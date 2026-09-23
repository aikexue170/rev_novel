"""Rev adapter: JevBench's three primitives map onto Rev's one `/score` question shape and back."""
from jevbench.adapters import RevHttpAdapter
from jevbench.adapters import rev_http as R
from jevbench.tasks import Task


def _task(qtype, criteria, labels, state={"a": 1}):
    return Task(id="t1", family="f", state=state, labels=labels, expected=labels[0], split="public",
                question={"type": qtype, "instructions": "Q?", "criteria": criteria})


def _run(monkeypatch, task, answer, extra=None):
    sent = {}

    def fake_post(url, body, headers, timeout):
        sent.update(url=url, body=body, headers=headers)
        return 200, {"answers": {"decision": answer}, "server_seconds": 0.05, "generated_tokens": 0, "path": "rows",
                     **(extra or {})}, 0.2

    monkeypatch.setattr(R, "http_post_json", fake_post)
    monkeypatch.delenv("REV_TOKENIZER", raising=False)
    return RevHttpAdapter(endpoint="http://127.0.0.1:8000/").run(task), sent


def test_choice_is_sent_as_is_and_read_natively(monkeypatch):
    r, sent = _run(monkeypatch, _task("choice", {"a": "A", "b": "B"}, ["a", "b"]),
                   {"choice": "b", "probabilities": {"a": 0.3, "b": 0.7}})
    assert sent["url"] == "http://127.0.0.1:8000/score"
    assert sent["body"] == {"state": {"a": 1}, "questions": [{"id": "decision", "instructions": "Q?",
                                                               "criteria": {"a": "A", "b": "B"}}]}
    assert "Authorization" not in sent["headers"]
    assert r.ok and r.probs == {"a": 0.3, "b": 0.7} and r.probs_source == "native" and r.label == "b"
    assert r.usage["output_tokens"] == 0 and r.usage["input_tokens_source"] == "estimate_chars_div_4"


def test_noul_true_false_maps_to_yes_no(monkeypatch):
    r, sent = _run(monkeypatch, _task("noul", {"true": "T", "false": "F"}, ["no", "yes"]),
                   {"choice": "true", "probabilities": {"true": 0.8, "false": 0.2}})
    assert sent["body"]["questions"][0]["criteria"] == {"true": "T", "false": "F"}
    assert r.ok and r.probs == {"yes": 0.8, "no": 0.2} and r.label == "yes"


def test_noul_without_criteria_gets_yes_no_texts(monkeypatch):
    _, sent = _run(monkeypatch, _task("noul", None, ["no", "yes"]),
                   {"choice": "false", "probabilities": {"true": 0.1, "false": 0.9}})
    assert sent["body"]["questions"][0]["criteria"] == {"true": "Yes", "false": "No"}


def test_score_levels_become_index_keys(monkeypatch):
    r, sent = _run(monkeypatch, _task("score", ["lo", "mid", "hi"], ["0", "1", "2"]),
                   {"choice": "2", "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}})
    assert sent["body"]["questions"][0]["criteria"] == {"0": "lo", "1": "mid", "2": "hi"}
    assert r.ok and r.probs == {"0": 0.1, "1": 0.2, "2": 0.7}


def test_wrong_keys_or_choice_are_failures_not_repairs(monkeypatch):
    r, _ = _run(monkeypatch, _task("choice", {"a": "A", "b": "B"}, ["a", "b"]),
                {"choice": "a", "probabilities": {"a": 1.0}})
    assert not r.ok and r.probs is None
    r, _ = _run(monkeypatch, _task("choice", {"a": "A", "b": "B"}, ["a", "b"]),
                {"choice": "z", "probabilities": {"a": 0.5, "b": 0.5}})
    assert not r.ok


def test_server_usage_is_preferred_for_token_count(monkeypatch):
    r, _ = _run(monkeypatch, _task("choice", {"a": "A", "b": "B"}, ["a", "b"]),
                {"choice": "a", "probabilities": {"a": 0.6, "b": 0.4}}, extra={"usage": {"input_tokens": 123}})
    assert r.usage["input_tokens"] == 123 and r.usage["input_tokens_source"] == "server"


def test_prompt_matches_server_template():
    text = R.rev_prompt("doc", "Q?", {"a": "A", "b": None})
    assert text == "State:\ndoc\nQuestion: Q?\nOptions:\na: A\nb: b\nDecision:"
    assert R.rev_prompt({"k": [1, 2]}, "Q?", {}).startswith('State:\n{"k":[1,2]}\n')


def test_self_hosted_reserves_nothing():
    assert RevHttpAdapter(endpoint="http://x").reserve_estimate(None) == 0.0
    assert RevHttpAdapter(endpoint="http://x", price_input_per_m=0.1).reserve_estimate(None) == 0.01
