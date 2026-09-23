# JevBench public items (231) vs Qwen3.8-27B (`dense27b`): analysis

Inputs: `jevbench_public_231.jsonl` (items in our request format), `dense27b.jsonl` (per-item results), `summary.json`
(scores + every leaderboard system on the same 231 items), `run.py` (build/score). Nothing here is training data;
the generator for the weak shapes is `/Users/rob/Documents/rev/synth_jevbench.py` (section 6).

## 1. Headline

| | easy (48) | standard (72) | hard (111) | pooled | Intelligence (public, item chance) | Calibration |
|---|---|---|---|---|---|---|
| ours (dense27b, 19,601 training rows) | 48/48 | 71/72 | 79/111 (71.2%) | 198/231 = 85.7% | **81.1** | 74.3 (ECE_hard 0.127, Brier 0.380, mean TVD on the 10 probability items 0.259) |
| simplejev-qwen3.8-27b (site rank 12) | 48 | 71 | 82 | 200 | 82.0 | site 81.1 |
| reflex-27b (rank 17) | 48 | 69 | 84 | 201 | 82.4 | site 86.2 |
| jev-1.13.0 (rank 1) | 48 | 71 | 81 | 200 | 82.3 | |
| untrained Qwen3.8-27B (Chutes, no fine-tune) | | | 47 | 166 | 63.0 | site 92.1 |
| gpt-5.6 / deepseek-flash (frontier, reasoning) | 48 | 70 | 107 | 225–226 | 96–97 | |

All 33 misses are in hard (32) or standard/adequacy (1). Easy and standard are solved (119/120), so the three
question-type conventions are not what is failing. Hard is where the whole leaderboard separates: the best
non-frontier system on these items gets 85/111; we get 79; frontier reasoning models get 107.

Intelligence arithmetic (from `run.py`): tier weights hard .30 / standard .28 / easy .14, renormalised over the three
tiers we can run (the judge tier has no public items), chance-corrected per tier. One hard item is worth
`0.30/0.72 × 100/(111 × (1 − 0.336)) ≈ 0.57` Intelligence points on this public-item metric. Standard and easy are
saturated, so every point of Intelligence we can still gain is a hard-tier item.

## 2. What the items look like, by tier and family

State is a string except where noted (dict states are JSON objects; the harness sends them as-is). Option keys are
snake_case labels with a one-sentence description each. Instructions are one sentence, often naming the document,
clause or record to apply ("Under PR-5 as amended by PR-5/A1, is application PRA-2026-02917 eligible …").

### easy (48, all correct)
| family | n | type | options | state chars (min/med/max) | shape |
|---|---|---|---|---|---|
| extraction | 12 | choice | 4–5 | 29/43/63 | one-sentence customer message; "Which payment method / size / priority does the customer name?"; keys are literal values (`paypal`, `XL`, `urgent`) |
| fact | 12 | noul | 2 | 48/59/81 | one or two sentences; "Has the order been shipped? Answer strictly from the facts stated in the text."; labels 6 yes / 6 no |
| intent | 12 | choice | 5 | 22/43/75 | one message; "Which intent does the user's message express?"; five intent keys |
| tool_selection | 12 | choice | 5 | 34/43/59 | one request; "Which single tool should be called …"; five tool-name keys |

### standard (72, 71 correct; items come in pairs `original-<family>-NN-{0,1}` sharing a group)
| family | n | type | options | state chars | shape and label distribution |
|---|---|---|---|---|---|
| adequacy | 12 | noul | 2 | 32/78/99 | `Request: … Response: …` in one line, optional reference; "Does the response fully satisfy the request, using the supplied reference when present?"; 6/6 |
| extraction | 12 | choice | 4 | 56/66/91 | short narrative with cancelled/hypothetical plans; "Extract the final confirmed delivery method … Choose unknown if …"; labels unknown 4, courier 4, pickup 2, post 2 |
| intent | 12 | choice | 5 | 50/61/82 | "Select the primary requested action. A mention without a request does not establish intent."; status 4, refund/change_address/cancel/other 2 each |
| ordinal | 12 | score | 4 | 45/71/109 | incident description; "Rate incident impact using only reported facts. Use the highest fully supported level."; levels 0:4, 1:2, 2:4, 3:2 |
| policy | 12 | noul | 2 | 83/124/158 | policy sentence + facts + request; "… Treat unproved required conditions as not satisfied."; 6/6 |
| routing | 12 | choice | 6 | 16/53/77 | one request; six specialist keys with a tie-break rule in the instruction; 2 per label |

### hard (111, 79 correct; groups `hard-{opus,sol}-{a,b,c}-<family>-NN`, i.e. two authoring pipelines: "opus" = long prose case packs, "sol" = compact rule-plus-facts or JSON records)
| family | n | types | options | state chars | shape |
|---|---|---|---|---|---|
| adversarial | 6 (6 ✓) | 3 choice, 2 noul, 1 score | 2–4 | 799/900/1025 | trusted metadata vs untrusted text (file previews containing "SYSTEM OVERRIDE …", auto-summaries); "Choose the access decision.", "Assign service priority from trusted evidence." |
| ambiguous | 7 (7 ✓) | choice | 3–4 | 4 (dict)/1288/1788 | conflicting evidence where one option is `cannot_determine` / `ask_user_which_alex`; two of seven expect the abstain option |
| judge_hard | 17 (12 ✓) | noul | 2 | dict {request, response} (~350–1000 chars) | "Does the response fully and correctly satisfy the request?"; criteria `true` = "fully and correctly satisfies every explicit request, including method, edge cases, units, formatting, and retained facts", `false` = "any substantive error or misses any explicit requirement"; labels 10 no / 7 yes |
| long_policy | 19 (12 ✓) | 12 choice, 5 noul, 2 score | 2–6 | 8051/12065/14986 | full policy pack: declarations, definitions, numbered clauses, exclusions, endorsements/amendments with effective dates, annexes, FAQ, worked examples, change log, superseded text kept "for reference", then a case file with trainee/clerk/handler notes; the question names the clause set to apply |
| multi_hop | 18 (14 ✓) | 16 choice, 1 noul, 1 score | 2–6 | 8 dict / 4834 / 10431 | either a handbook with catalog + matrix + directory + rotation + override tables and an alert to route, or a JSON record with `archive` (15 distractor records each carrying a "review_note"), `*_map` tables, `footnotes`, and a `calculation_rule` with a strict boundary |
| probability | 10 (8 ✓) | 7 choice, 3 noul | 2–3 | 1073/2100/4946 | frequency table or shipment history; "Give probabilities that reflect the evidence"; `gold_probs` supplied (Calibration fidelity term); answer = argmax after conditioning/filtering |
| routing_hard | 5 (5 ✓) | choice | 5 | dict {handlers, request} | five handler descriptions in the state and in the criteria; request mixes a symptom with a misleading self-diagnosis |
| temporal_numeric | 15 (2 ✓) | 10 choice, 4 noul, 1 score | 2–6 | 2 dict / 1474 / 2780 | "opus" items: a rule set (warranty term, alert rule, dosing rule, working-time rule, cancellation terms, facility agreement, card schedule, travel policy) plus a record (daily export, roster, folio, activity list) and a human note that computes it wrongly; choice keys are the candidate amounts/days (`eur_692_00`, `sep_26`); "sol-b" items: JSON {event, rules} with "Apply all stated timing, unit, inclusion, correction, and threshold rules; select the exact band." |
| tradeoff | 6 (5 ✓) | 4 choice, 2 noul | 2–5 | 1629/1923/2533 | ranked precedence rules with definitions that override surface cues, or ordered dispatch rules, or a control checklist |
| trap | 8 (8 ✓) | 4 noul, 3 choice, 1 score | 2–4 | 515/887/1251 | a tempting cue (customer says "unopened", a note says "authorized for override", telemetry is clean) that the operative rule defeats; criteria `true` = "The stated condition is satisfied under the operative facts." |

## 3. The three question types: exact conventions

Our training data (`data/train.jsonl.gz`, 19,792 rows) is essentially all `choice`: criteria dicts keyed by label or
index, e.g. `{"yes": "Correct answer", "no": "Incorrect answer"}` (MultiRC/BoolQ), `{"0": …, "1": …}` (CosmosQA/RACE),
`{"1","2","3"}` (Social IQa), 77-way Banking77. The benchmark's other two types reach our server like this
(`run.py::convert`):

- **noul**: criteria are keyed `"true"` and `"false"`, sent as-is. `true` carries the positive description ("The
  condition is established by trusted facts under the stated rule."), `false` the negative. The server's chosen key is
  mapped `true → yes`, `false → no`; the harness scores over `{"yes","no"}` and takes p_yes from our `true`
  probability. Descriptions are always one sentence and always spell out the disjunction on the `false` side ("Wrong,
  incomplete, unsupported or violates a constraint"). Labels across the 74 public noul items: 38 no / 36 yes; in
  judge_hard 10 no / 7 yes.
- **score**: the benchmark gives an ordered list of level descriptions; we send `{"0": level0, "1": level1, …}` and the
  expected label is the level index as a string. Levels are monotone ("SEV-4 (lowest) … SEV-1 (highest)", "No
  violations / Exactly one / Exactly two / Three or more", "Level 0 — no award … Level 4 — EUR 3,000"). Instructions
  often say "Output the level index." Public score items: standard/ordinal 12 (levels 0:4, 1:2, 2:4, 3:2) and 6 in
  hard (expected 0, 1, 1, 1, 2, 2).
- **choice**: keys are the benchmark labels, snake_case, sometimes value-bearing (`eur_498_81`, `sep_26`,
  `pay_subject_to_15000_sublimit`, `tier3_cfo`). Descriptions restate the outcome, and in tradeoff/trap items they name
  the rule that would justify it ("R3 is the highest-ranked applicable rule: …").

Ties: `run.py` picks the lexicographically smallest label among tied maxima, as the harness does.

**Is the format hurting us?** Not on its own: noul 12/12 easy, 23/24 standard; score 12/12 standard. The
`true/false → yes/no` mapping and the level-index convention are learned already (from the earlier
`synth_workflows.py` rows, which use `true`/`false` keys, and from BoolQ/MultiRC). By type on hard: choice 48/67,
noul 29/38, score 2/6. Hard score is bad (2/6), but all four hard score misses are off by exactly one level in the
same direction (+1) after long counting/adjustment chains, which is arithmetic, not level-format confusion.

## 4. Every miss, with a one-line diagnosis

Confidence = probability on the predicted (wrong) label. "Followed note" = the wrong answer equals the conclusion of
a trainee/chat/tool/customer note inside the state.

| # | item | type | expected → predicted (conf) | diagnosis |
|---|---|---|---|---|
| 1 | original-adequacy-05-0 | noul | no → yes (.81) | constraint check: "Say exactly two words" vs a three-word reply; did not count |
| 2 | hard-opus-a-long_policy-01 | choice | 15k sublimit → 10k (.70) | endorsement applicability by renewal date (term renewed after 1 Jan 2026 → $15,000); used the base-form figure |
| 3 | hard-opus-a-long_policy-04 | choice | cfo → vp_and_finance_director (.81) | TCV = 3 × 41,500 + 9,800 + 2,400 = 136,700, conditional discount disregarded, plus 14,000 related purchase within 90 days = 150,700 > 150,000; missed the aggregation clause (boundary by 700); followed the requester's "annual cost is only 41.5k" framing |
| 4 | hard-opus-a-long_policy-19 | noul | no → yes (.67) | excluded income: the December bonus must be removed from Baseline (6-month amended window) → drop 23.1% < 25%; followed the handler's draft which kept the bonus (28.8%) |
| 5 | hard-opus-a-probability-04 | choice | on_time → late (.62) | filtering: Brisa+Express excluding peak weeks 47–52 gives 9 on time / 5 late / 1 early; followed the planner's "12 late of 24" (unfiltered) |
| 6 | hard-opus-a-temporal_numeric-03 | noul | yes → no (.86) | date math: 18 months from 31 Aug 2026 → Feb 2028 has no 31st → last day = 29 Feb (leap year); 18:30 UTC−4 = 23:30 CET, before 24:00; followed the claims-centre note ("lands on 28 Feb") |
| 7 | hard-opus-a-temporal_numeric-07 | choice | sep_26 → sep_24 (.45) | running sum over 26 rows of net cost (usage − credit, tax excluded) reaching 80% of 12,000; followed the FinOps note that included tax |
| 8 | hard-opus-a-temporal_numeric-09 | noul | no → yes (.66) | time zones: 13:00 CDT = 18:00 UTC; 01:30 CEST = 23:30 UTC → 5.5 h elapsed, not 12.5; followed the patient's clock arithmetic |
| 9 | hard-opus-a-temporal_numeric-12 | score | 1 → 2 (.39 vs .36) | DST: S4→S5 is 10.5 h by the clock but 11.5 h elapsed (clocks back); only S1→S2 (10.5 h) violates; near-tie, so calibration was honest |
| 10 | hard-opus-b-multi_hop-04 | choice | bjorn → esra (.39) | 6-hop chain: ledgerd → ledger-core → Payments-Core (merged into Money Movement) → tier-1 CrashLoop = SEV2 → 06:30Z = 08:30 Berlin, week of 09-14 → Ana → override 09-20 18:00–09-21 09:00 replaces Ana with Bjorn; picked the override that starts at 09:00 (time-zone/boundary slip); the chat note pointed at Dmitri |
| 11 | hard-opus-b-multi_hop-05 | score | 2 → 3 (.51) | civil partnership = married → independent; household 2; countable 7,200 + 20,600 = 27,800 (work-study and small gift excluded) → 13,900 → Level 2; the model seems to have dropped the student's own job as well (20,600/2 = 10,300 → Level 3); trainee note said Level 1 |
| 12 | hard-opus-b-probability-02 | choice | upstream_provider → bad_push (.83) | conditioning: deploy at 09:52, alert 14:07 → no deploy in prior 2 h → column 4/13/3 → upstream; followed the on-call's base-rate ("60% of the time") |
| 13 | hard-opus-b-tradeoff-01 | choice | escalate_to_engineering → escalate_to_security (.86) | definition lookup: tn_9120 shares the Org ID → same organization → R1 does not apply; R3 (reproducible defect, Enterprise) does; fell for the "somebody else's data" cue |
| 14 | hard-opus-c-long_policy-03 | choice | eur_400 → eur_200 (.52) | minute-level definition: Arrival Time = door open 21:14Z vs scheduled 18:10Z → 3 h 04 ≥ 3 h; distance band 1,500–3,500 → 400; E2 halving needs delay ≤ 3 h, which it exceeds; technical defect not extraordinary under Revision 3; followed the desk pre-fill ("under 3 h", "F5 suggested YES") |
| 15 | hard-opus-c-long_policy-05 | score | 1 → 2 (.48) | 8-part count: 96 − 2 duplicates + 3 recent leavers + 6 kiosk users + 8 contractors + 1 intern via service account + 5 QA testers (tenant registered after the date); parental leave 77 days < 90 so counted → 117 vs 115 held → shortfall 2; accumulated one bucket too many |
| 16 | hard-opus-c-long_policy-08 | score | 1 (SEV-3) → 2 (SEV-2) (.87) | ordered adjustments: 318/4,020 = 7.9% → SEV-3; only 2 Enterprise (trial excluded); duration exactly 4 h (13:40–17:40) is not "more than"; workaround from 14:08 lowers to SEV-4; Enterprise floor raises to SEV-3; followed the IC's "keep SEV-2" |
| 17 | hard-opus-c-long_policy-10 | choice | credit_minus_restocking_fee → reject_return (.57) | date + definitions: delivered 12 Aug, RMA 10 Sep = 29 days ≤ 30; catalogue codes only → Standard (Amendment 1); NFF → not Defective; 31 powered h → Unused; Silver on RMA date → 15% fee; model rejected (likely counted from invoice date or treated NFF as ineligible) |
| 18 | hard-opus-c-temporal_numeric-02 | noul | yes → no (.92) | DST + zones: Sunday Berlin 01:00–12:00 spans the spring-forward hour → 10 h; week total 9+9+8+2+10+10 = 48 ≤ 48; rest Sat 09:00 NY → Sun 01:00 CET = 11 h; followed the tool's "49.0"; most over-confident temporal miss |
| 19 | hard-opus-c-temporal_numeric-03 | choice | eur_692_00 → eur_617_00 (.34) | leap year + fee window: notice 10 Feb 2028 + 30 = 11 Mar (Feb has 29 days); term 366 days; remaining 173 → 1,464 × 173/366 = 692.00; six months ended 29 Feb so no EUR 75 fee; picked 692 − 75 |
| 20 | hard-opus-c-temporal_numeric-04 | choice | usd_2335_00 → usd_2303_01 (.48) | segmented act/360 interest: 17 d × 250k @ 4.20 + 29 d × 200k @ 4.20 + 45 d × 200k @ 4.65 (reset = 2nd business day after 28 Feb = 1 Mar); pure arithmetic |
| 21 | hard-opus-c-temporal_numeric-06 | choice | sep_18_book_set → sep_16_taxi (.38) | running available limit with holds, 3% foreign fee, partial refund not restoring, 8-day release: 4.50 left before 18 Sep → book set declined; pure sequential computation |
| 22 | hard-opus-c-temporal_numeric-08 | choice | eur_498_81 → eur_497_40 (.65) | FX rate-date rule (folio charged Sun 11 Oct → Fri 9 Oct rate 162.40), per-night cap 170, taxes in full, breakfast out: 162.56 + 162.56 + 170 + 3 × 1.23 = 498.81; near-miss options within EUR 35 |
| 23 | hard-sol-a-multi_hop-08 | choice | central_engineering → local_maintenance (.76) | strict boundary: 38 − 8 = 30 and "exactly 30 is not under" |
| 24 | hard-sol-a-multi_hop-10 | choice | approve_45_days → reduce_to_30_days (.80) | footnote chain: A2 max 30^9; footnote 9 → 60 days for RPT when all fields non-granular; revision 4 moved "region" to non-granular; missed the footnote |
| 25 | hard-sol-b-judge_hard-02 | noul | no → yes (.99) | verification: response says 71.5 + 12 = 84.5 (correct is 83.5); did not re-add |
| 26 | hard-sol-b-judge_hard-08 | noul | no → yes (.91) | rounding: 23.22576 → 23.226, response truncated to 23.225 |
| 27 | hard-sol-b-judge_hard-10 | noul | no → yes (.95) | business-day count with two holidays: 8th day is Mon 8 Dec; response skipped it and said Tue 9 Dec |
| 28 | hard-sol-b-temporal_numeric-01 | choice | late_by_under_2h → within_window (.51) | elapsed window: opens Mon 9 Mar 09:00 (first business day after Friday-evening notice), 52 h → Wed 13:00; reply 13:30 → 30 min late |
| 29 | hard-sol-b-temporal_numeric-02 | choice | over_limit → under_100gb_below (.41) | unit conversion: 2.40 TiB = 2,638.83 decimal GB; 875.50 + 1,120.25 − 14.75 + 670.00 = 2,651.00 > allowance |
| 30 | hard-sol-b-temporal_numeric-03 | choice | 9_credits → 12_credits (.61) | inclusive day count with an exclusion: Feb 28, Mar 1, Mar 2 active (Feb 29 suspended) = 3 × 3 = 9; counted 4 days |
| 31 | hard-sol-b-temporal_numeric-04 | choice | pass → clear_fail (.50) | kg→g and percent: (1.65 + 0.20)/2,500 g = 0.074% ≤ 0.080% |
| 32 | hard-sol-c-judge_hard-07 | noul | no → yes (.70) | unit factor: 1 g/cm³ = 1000 kg/m³, response used 100 |
| 33 | hard-sol-c-judge_hard-13 | noul | no → yes (.93) | half-up rounding: 18.35649375 → 18.36, response gave 18.35 |

### Patterns across the 33

- **Reasoning vs format/convention.** 0/33 are label-set or convention failures (the model never emits an invalid
  distribution, never confuses level indices, never flips true/false on easy/standard). The one family where a
  prior rather than a computation drives the error is judge_hard: all 5 misses are expected `no`, predicted `yes` at
  .70–.99, i.e. the model rates a fluent, well-formatted derivation as correct without re-doing the arithmetic. That is
  partly a data prior (our MultiRC/BoolQ rows reward "plausible = yes") and partly a capability gap, so at most
  **5/33 ≈ 15% are convention/prior-driven; ≥ 85% are reasoning misses**. Within reasoning: 13 are multi-step
  arithmetic / running computations, 9 involve dates, time zones, DST or leap years (overlapping), 7 are policy
  reading (amendment applicability, definitions overriding surface cues, footnotes, conditioning), 1 is a strict
  boundary, 3 are verification of rounding/units in a judge role, 1 is a word-count constraint.
- **Distractor-following.** In 14/33 misses (42%) the wrong answer is exactly the conclusion of a note inside the
  state (trainee draft, chat, tool pre-fill, customer claim). The public "opus" items nearly always contain such a
  note, and it is wrong roughly two thirds of the time. The model treats the note as evidence instead of recomputing.
- **Direction of score errors.** All four hard score misses are +1 level (over-counting / over-severity).
- **Calibration.** Median top-probability on hard hits is 0.97, on misses 0.65; nine misses are ≥ 0.85, seven of them
  in judge_hard, tradeoff-01, long_policy-08 and temporal-02, where a distractor made the wrong answer look certain.
  ECE_hard 0.127 and Brier 0.380 come mostly from those. Mean TVD on the 10 probability items is 0.259: even where the
  argmax is right the distribution is far from the gold frequencies (the model outputs near-one-hot probabilities).
- **The untrained base scored 7/15 on temporal_numeric (site: hard 21% overall); our fine-tune scores 2/15.** The
  training mix (reading-comprehension QA, no arithmetic, no dates) appears to have moved the model toward
  "read the human's number" behaviour. This is the strongest argument for adding computation-heavy synthetic data.

## 5. Learnable from synthetic data vs needs real capability

| family | verdict | why |
|---|---|---|
| temporal_numeric | **learnable in part, biggest upside (13 misses)** | every sub-shape is a closed-form rule (month-end clamp, zone offsets, DST, act/360, running limits, FX rate-date, pro-rata with fee windows, unit conversion). Labels are exact by construction, and the "wrong human note" can be generated by running the naive method. Whether a 27B *classifier* (no chain of thought) can learn to do 26-row running sums in the forward pass is the open question; near-boundary arithmetic (misses 3, 9, 19, 20, 22) may stay hard. Expect 2/15 → 6–9/15 |
| long_policy | **learnable (7 misses)** | the misses are not about length per se; they are amendment applicability by date, excluded-income definitions, ordered adjustments with floors, and multi-bucket counts, all rule-derivable. Synthetic packs of 5.8–10k chars (median 8.2k) with the same skeleton (definitions → clauses → amendment → annex → case file → wrong draft) target exactly this. Expect +2–4 of 7 |
| multi_hop | **learnable (4 misses)** | on-call chains, bursary assessments and JSON registry/footnote chains are generated with the answer traced through tables; the strict-boundary miss (#23) and the footnote miss (#24) are the cheapest to fix |
| judge_hard | **learnable for the arithmetic/rounding/unit/business-day cases (all 5 misses)** | request/response pairs with injected errors are trivial to generate with balanced yes/no. The judge tier on the site (146 items, weight .28, not public) is the same shape at lower difficulty, so this also protects the hidden tier |
| tradeoff, routing_hard, trap, adversarial | learnable and mostly already solved (1 miss) | keep a modest amount so the retrain does not regress the definition-over-surface-cue behaviour |
| probability | learnable for the argmax (2 misses: conditioning, filtering) | for the Calibration fidelity term the model would need to output frequency-shaped distributions; a classifier trained with one-hot labels will not learn that from our row format unless we add soft targets (our rows have `expected` only; `eval_sets/jev_official_262` has `target_distribution`, so the format exists) |
| ambiguous | not synthesised | 7/7 correct; the abstain option depends on genuinely irreconcilable evidence, which is hard to generate without templating the answer |
| adequacy (standard) | learnable (1 miss) | explicit-constraint checking (word/sentence/item counts, decimals, case) is rule-generable |

The remaining gap to the frontier models (107/111) is real capability: they get 14/15 temporal and 18–19/19
long_policy with reasoning at inference. Without chain of thought we should expect to close roughly half the gap on
the rule-derivable families and little of it on the near-boundary arithmetic.

## 6. The generator: `synth_jevbench.py`

`python3 synth_jevbench.py OUT_DIR [--n 500] [--seed 7] [--families …]` writes `train.jsonl` / `dev.jsonl` (10% of
groups) in the training row format `{id, group, source, state, instructions, criteria, expected, upstream_split}` and
`summary.json`. `python3 synth_jevbench.py OUT_DIR --check` asserts zero exact / 120-character-prefix / normalised-prefix
overlap of states, instructions **and option descriptions** with the 231 public items and `eval_sets/*.jsonl`, and
prints per-family label balance, question types and state lengths.

Nine families, 41 sub-generators, labels by rule in every case. The "naive" human note in a state is produced by
running the wrong method (wall-clock instead of elapsed, tax included, bonus kept, base form instead of amendment),
so it is wrong exactly when the benchmark's notes are wrong, and it is replaced by a neutral reminder ~30% of the
time so the model cannot learn "always contradict the note".

| family | sub-generators | types |
|---|---|---|
| long_policy | procurement approval level (TCV, renewals, conditional vs unconditional discounts, aggregation window with amendment), returns desk (delivery vs invoice date, X-codes, NFF, unused test, tier on RMA date), insurance sublimit (endorsement applicability by renewal date, vacant vs unoccupied, concealed vs known seepage), licence true-up (8 inclusion/exclusion buckets, order-form start dates), payment relief (excluded bonus, amended baseline window, 24-month start-date rule, DPD/affordability boundaries), incident severity (base, payment/data floors, ordered duration/workaround adjustments, Enterprise floor) | choice, choice, choice, score, noul, score |
| temporal_numeric | warranty end (month-end clamp, leap years, two zones), budget alert (running net sum, credits, tax excluded), dose interval across zones, rest violations across DST (score), weekly rota compliance across DST/zones, pro-rata refund (leap-year term, fee window), loan interest (act/360, prepayment value date, 2nd-business-day reset with holidays), card declines (holds, 8-day release, 3% fee, partial vs full refunds), FX lodging (rate-date rule, per-night cap, taxes), JSON {event, rules} bands (elapsed windows over DST, TiB vs GB, inclusive day proration with suspension, percent with uncertainty) | noul, choice, noul, score, noul, choice, choice, choice, choice, choice |
| multi_hop | on-call routing (process → service → team → merge → severity matrix → rotation week → local time → override [start,end)), bursary (status → household → countable income with exclusions → per-capita → table → asset cap), JSON registry chains (alias → registry → matrix → footnote → strict boundary, with 5–9 archive distractor records) | choice, score, choice |
| routing_tradeoff | ranked precedence rules with definitions (same Org ID, Trial ≠ Enterprise, reproducible steps), handler selection (3 domains × 5 handlers with misleading self-diagnosis), dispatch slot under ordered rules, §7 control checklist | choice, choice, choice, noul |
| trap_adversarial | access decision from trusted metadata vs untrusted file text, priority from telemetry vs customer claims (score), ticket disposition under a closure rule, condition-established-by-trusted-facts (paid promotion, unopened seal, endorsement in chat, marketing consent) | choice, score, choice, noul |
| noul_yesno | policy permitted with unproved conditions, facts strictly stated (negations, plans, hypotheticals, "now" updates), adequacy of a response to explicit constraints (word/sentence/item counts, decimals, case, question mark, reference) | noul |
| score_levels | incident impact (highest fully supported level, with unconfirmed speculation), breach counts with unit conversions (minutes, lb→kg, °F→°C) and inclusive limits, shortfall bands after adjustments | score |
| judge_verify | 9 request kinds (tank volume, area conversion, business days with holidays, density g/cm³→kg/m³, discount-then-tax with half-up, percent change, km→miles, mean, shift length) × 6 error injections (arithmetic slip, wrong rounding, unit factor, missing format constraint, wrong order, none) | noul, balanced 50/50 |
| probability_freq | conditional table (deploy within 2 h / weekend) with the unconditioned argmax as the chat distractor; filtered shipment history (carrier + service, peak weeks excluded) | choice |

Layouts: policy packs are numbered prose with 9–13 boilerplate sections drawn from a pool of 22 (record keeping,
evidence standard, counting periods, version control, …), worked examples, FAQ, change log, glossary/roles/template
annexes, then a case file rendered as JSON, YAML-ish, key: value prose or a pipe table; records in the other families
are JSON objects, markdown tables, aligned exports, or prose. Instructions draw from 3–4 phrasings per sub-generator;
criteria descriptions vary; entities, companies, suppliers, cities, dates and numbers are all sampled.

Sample run (`--n 500`, seed 7) is at
`/private/tmp/claude-501/-Users-rob-Documents-ChatGPT-Jev/00778d75-7b56-4974-b1cd-e1de4841c85d/scratchpad/synth_jevbench_sample/`
with its `--check` output in `check_output.txt` (zero overlap; balance printed per family).

## 7. Recommendation

**Which families and how many rows for a retrain** (on top of the existing 19.6k; the synthetic block should be ~30%
of the mix so the reading-comprehension base is not drowned):

| family | rows | rationale |
|---|---|---|
| temporal_numeric | 2,000 | 13 of 33 misses; the capability we actively lost in fine-tuning (base 7/15 → ours 2/15) |
| long_policy | 1,200 | 7 misses; long contexts also teach "ignore the draft, apply the amendment"; rows are 6–10k chars so this is the bulk of tokens |
| judge_verify | 1,000 | 5 misses, all over-confident `yes`; balanced 50/50 directly attacks the yes-prior and protects the hidden judge tier |
| multi_hop | 800 | 4 misses; boundary and footnote habits generalise |
| noul_yesno + score_levels | 600 + 400 | keep the true/false and level-index conventions dense in the mix; cheap short rows |
| routing_tradeoff + trap_adversarial | 400 + 400 | already solved; enough to prevent regression |
| probability_freq | 300 | 2 misses; optionally emit `target_distribution` from the same frequency tables if the trainer supports soft labels (this is the only lever on the Calibration fidelity term) |

Hold out the generator's `dev.jsonl` and, separately, keep the 231 public items as a test only (they must not enter
training).

**27B, 4B or both.** Retrain the 27B first: it is the arm we can measure (this run), it has the headroom on exactly
the rule-derivable families, and the base model demonstrably had more temporal ability than the fine-tune retained.
Retrain the 4B with the same mix afterwards; there is no 4B run on these 231 items in the inputs, so its per-family
profile is unknown, but on the leaderboard every 4B-class system sits at 33–41% hard (semif-qwen3.5-4b: 3/15
temporal, 10/19 long_policy) and gains there will come more from trap/routing/judge shapes than from arithmetic.
Do both only if the 27B retrain shows the temporal family moving; if it does not move on the 27B it will not move on
the 4B.

**Expected effect on Intelligence (public-item metric, each hard item ≈ +0.57).** Conservative: temporal +4,
long_policy +2, judge_hard +1, multi_hop +1 → hard 87/111 → Intelligence ≈ 84 (from 81.1), above every non-frontier
system on these items. Optimistic: temporal +7, long_policy +3, judge +2, multi_hop +2, tradeoff/probability +1 →
94/111 → ≈ 89. Regression risk is on standard/easy (currently 119/120) if the synthetic share is too high, and on
ambiguous (7/7) which the generator does not cover; watch both. Calibration should improve as a side effect because
the over-confident misses are concentrated in the judge_verify-shaped items; the TVD term will not move without soft
targets.
