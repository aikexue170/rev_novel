#!/usr/bin/env bash
# Full Decision Index run for Rev, end to end. Needs: an HF login (the frozen suite dataset is gated/private:
# `hf auth login` or HF_TOKEN=...), the kit checked out, and a running Rev server (our Modal endpoint, or
# serve_local.py). Wall time at concurrency 8 against our Modal 27B: ~132k requests at the ~10 req/s we measured on
# the sample is ~3.7 h; ToolRet/BRIGHT/ACOS rows carry 30+ questions each, so budget 4-6 h. The kit's own serial
# runner (`decision_index run`) would take ~25-30 h at ~0.7 s/request; use run_parallel.py instead (same output).
set -euo pipefail
: "${HF_TOKEN:?set HF_TOKEN (or run: hf auth login)}"
export HF_HUB_DISABLE_XET=1
REV=${REV:-/Users/rob/Documents/rev}                       # this checkout
KIT=${KIT:-$HOME/decision-index}                           # git clone https://github.com/apolinario/decision-index
ARM=${ARM:-rev-27b}
URL=${URL:-https://reprompt--jev-beat-serve-dense27jb.us-west.modal.direct}
CONC=${CONC:-8}
export PYTHONPATH=$REV:${PYTHONPATH:-}

[ -d "$KIT" ] || git clone https://github.com/apolinario/decision-index "$KIT"
pip install -q -e "$KIT" httpx tokenizers                  # kit + adapter deps (no torch needed for the HTTP backend)
cd "$KIT"

# 1. frozen suite (132,422 rows; verified against the pinned sha256 750d353a...)
python -m decision_index suite download --dir suite
python -m decision_index suite verify --dir suite

# 2. run (resumable: rerun the same command to retry rows whose status is error)
python "$REV/decision_index_adapter/run_parallel.py" --rows suite/selected-rows.jsonl.gz --out "runs/$ARM" --base-url "$URL" --concurrency "$CONC"

# 3. score: benchmark-summary.json, index.json, scores.json (scores.json must say "complete": true)
python -m decision_index score --results "runs/$ARM/results.jsonl" --suite-dir suite --engine "$ARM"
python - <<EOF
import json; s=json.load(open("runs/$ARM/scores.json")); print("Decision Index", s["decision_index"], "complete", s["complete"], "counts", s["counts"])
EOF

# 4. upload for the submission PR (private dataset; add --public to expose it)
#    python -c "from decision_index.pipeline import upload_run; print(upload_run('robbalian/decision-index-results', 'runs/$ARM', path_in_repo='runs/$ARM'))"
