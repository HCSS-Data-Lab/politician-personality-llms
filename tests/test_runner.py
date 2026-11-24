"""Offline behavioral checks: no API client, account, GPU or credentials required."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_runner", ROOT / "scripts" / "run_models.py")
runner = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


@pytest.fixture
def sample():
    return [
        {"leader_id": "person-a", "full_name": "Alex Example", "identity": "Alex Example, President of Exampleland", "country": "Exampleland"},
        {"leader_id": "person-b", "full_name": "Blair Example", "identity": "Blair Example, President of Exampleland", "country": "Exampleland"},
    ]


@pytest.fixture
def speeches():
    return [{"leader_id": "person-a", "speech_id": "a.txt"}, {"leader_id": "person-b", "speech_id": "b.txt"}]


class FakeClient:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = iter(responses) if responses is not None else None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        value = next(self.responses) if self.responses is not None else json.dumps({f"s{i}": 2 for i in range(1, 17)})
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=value))])


def test_full_executed_matrix(sample, speeches):
    tasks = runner.plan(sample, speeches, "qwen-8b", recognition=True)
    assert len(tasks) == 62
    assert len({t.identity for t in tasks}) == len(tasks)
    for condition, framings in runner.MATRIX.items():
        selected = [t for t in tasks if t.condition == condition]
        assert {t.framing for t in selected} == set(framings)
        assert {t.repeat for t in selected} == {0, 1, 2}
    probes = [t for t in tasks if t.condition == "recognition"]
    assert len(probes) == 2 and all(t.repeat == 0 for t in probes)
    assert not any(t.condition == "anon-speech" and t.framing == "P" for t in tasks)


def test_plan_rejects_invalid_manifest(sample, speeches):
    with pytest.raises(ValueError, match="Duplicate sample"):
        runner.plan(sample + sample[:1], speeches, "qwen-8b")
    with pytest.raises(ValueError, match="filename"):
        runner.plan(sample, [{"leader_id": "person-a", "speech_id": "../secret"}], "qwen-8b")
    with pytest.raises(ValueError, match="association"):
        runner.plan(sample, speeches + speeches[:1], "qwen-8b")


def test_dry_run_does_not_read_speeches_create_client_or_write(tmp_path, sample, speeches):
    tasks = runner.plan(sample, speeches, "qwen-8b", recognition=True)
    def forbidden():
        pytest.fail("Dry-run must not construct a client")
    result = runner.run(tasks, sample, root=tmp_path, backend="vllm", base_url="dry-run",
                        output=tmp_path / "outputs" / "trial", client_factory=forbidden)
    assert result["dry_run"] and result["executed_requests"] == 0
    assert list(tmp_path.iterdir()) == []


def test_no_fresh_output_can_overwrite_frozen_records(tmp_path, sample):
    with pytest.raises(ValueError, match="below outputs"):
        runner.run([], sample, root=tmp_path, backend="vllm", base_url="dry-run", output=tmp_path / "records")


def test_word_cap_preserves_whitespace():
    assert runner.cap_words(" one\n two  three four  ", 2) == " one\n two"
    assert runner.cap_words(" one\n two  ", 2) == " one\n two  "
    assert runner.cap_words("anything", 0) == ""
    with pytest.raises(ValueError):
        runner.cap_words("anything", -1)


def test_evidence_blocks_and_prompt_parity(tmp_path, sample):
    for kind in ("clean", "anon"):
        path = tmp_path / "data" / "speeches" / kind
        path.mkdir(parents=True)
        (path / "a.txt").write_text("This is an invented test speech.", encoding="utf-8")
    person = sample[0]
    for frame, original in runner.FRAMINGS.items():
        task = runner.Task("person-a", "qwen-8b", "name-context", frame, 1)
        assert runner.build_prompt(task, person, tmp_path) == runner.I.build_prompt(person["identity"], original, 1)
        task = runner.Task("person-a", "qwen-8b", "named-speech", frame, 1, "a.txt")
        prompt = runner.build_prompt(task, person, tmp_path)
        assert prompt == runner.I.build_named_speech_prompt(person["identity"], "This is an invented test speech.", original, 1)
    bare = runner.build_prompt(runner.Task("person-a", "qwen-8b", "name-only", "neutral", 0), person)
    context = runner.build_prompt(runner.Task("person-a", "qwen-8b", "context-only", "neutral", 0), person)
    assert "Public figure: Alex Example\n" in bare
    assert "Public figure: President of Exampleland\n" in context
    assert "Alex Example" not in context
    anon = runner.build_prompt(runner.Task("person-a", "qwen-8b", "anon-speech", "neutral", 0, "a.txt"), person, tmp_path)
    assert "Alex Example" not in anon and "Public figure:" not in anon


def test_source_prompt_golden_hashes(sample):
    # Independently computed against the original executed instrument, using
    # invented evidence only. These detect wording/order drift in a copied module.
    cases = [
        (runner.Task("person-a", "qwen-8b", "name-context", "E", 0), "95976c5bc744df524b000ed3f509607c1d508d0b3e4a59f47707c8c1b1debdce"),
        (runner.Task("person-a", "qwen-8b", "name-context", "neutral", 1), "b3a20db67d0c8d0292dc0267c9229aed6cfa15d702fe6bce4c705187030b94aa"),
        (runner.Task("person-a", "qwen-8b", "name-context", "P", 2), "cedca87e628164fa8ad8436d0178927e1c3e1c5d38b679c3384766a8855a2a4b"),
        (runner.Task("person-a", "qwen-8b", "context-only", "neutral", 2), "b1f1de6503780b22e83c6c4fb8786c6e377f551a9b187cbe122b7fabf9f6bb37"),
    ]
    for task, expected in cases:
        assert hashlib.sha256(runner.build_prompt(task, sample[0]).encode()).hexdigest() == expected


def test_request_settings_have_original_schema_and_thinking_controls():
    open_task = runner.Task("person-a", "qwen-8b", "name-context", "neutral", 1)
    options = runner.request_options(open_task, "invented prompt", "vllm")
    assert options["temperature"] == .7 and options["seed"] == 1 and options["max_tokens"] == 512
    assert options["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert options["extra_body"]["response_format"]["json_schema"] == runner.I.RATINGS_SCHEMA
    closed_task = runner.Task("person-a", "gpt-5.5", "name-context", "neutral", 1)
    options = runner.request_options(closed_task, "invented prompt", "openrouter")
    assert "temperature" not in options
    assert options["extra_body"]["reasoning"] == {"enabled": False}
    assert "chat_template_kwargs" not in options["extra_body"]
    gemma = runner.request_options(runner.Task("person-a", "gemma-12b", "recognition", "neutral", 0, "a.txt"), "invented", "vllm")
    assert "chat_template_kwargs" not in gemma["extra_body"]
    assert gemma["extra_body"]["response_format"]["json_schema"] == runner.I.REID_SCHEMA


def test_duplicate_contexts_share_requests_keep_associations_and_resume(tmp_path, sample):
    tasks = runner.plan(sample, [], "qwen-8b", ["context-only"])
    client = FakeClient()
    dest = tmp_path / "outputs" / "trial"
    report = runner.run(tasks, sample, root=tmp_path, backend="vllm", base_url="https://example.invalid/v1",
                        output=dest, execute=True, client_factory=lambda: client)
    assert len(client.calls) == 3
    rows = [json.loads(line) for line in (dest / "ratings.jsonl").read_text().splitlines()]
    assert len(rows) == 6 and {r["leader_id"] for r in rows} == {"person-a", "person-b"}
    assert report["written_associations"] == 6
    def forbidden():
        pytest.fail("A resumed complete plan must not construct a client")
    report = runner.run(tasks, sample, root=tmp_path, backend="vllm", base_url="https://example.invalid/v1",
                        output=dest, execute=True, client_factory=forbidden)
    assert report["written_associations"] == 0
    changed = [dict(sample[0], identity="Alex Example, Another office"), sample[1]]
    with pytest.raises(ValueError, match="different or unverified request"):
        runner.run(tasks, changed, root=tmp_path, backend="vllm", base_url="https://example.invalid/v1",
                   output=dest, execute=True, client_factory=forbidden)
    with pytest.raises(ValueError, match="incompatible"):
        runner.run(tasks, sample, root=tmp_path, backend="vllm", base_url="https://other.invalid/v1",
                   output=dest, execute=True, client_factory=forbidden)


def test_parse_retry_bumps_sampling_seed_but_keeps_prompt(tmp_path):
    ratings = json.dumps({f"s{i}": i % 5 for i in range(1, 17)})
    client = FakeClient(["not JSON", ratings])
    task = runner.Task("person-a", "qwen-8b", "anon-speech", "E", 2, "a.txt")
    value = runner.obtain(client, task, "invented prompt", "vllm", "https://example.invalid/v1", tmp_path)
    assert value["s1"] == 1
    assert [r["seed"] for r in client.calls] == [2, 100002]
    assert client.calls[0]["messages"] == client.calls[1]["messages"]
    assert next(iter(tmp_path.glob("*.json"))).read_text().find("invented prompt") < 0
    second = runner.obtain(FakeClient([]), task, "invented prompt", "vllm", "https://example.invalid/v1", tmp_path)
    assert second == value


def test_transport_retry_and_failure_redaction(tmp_path):
    class Retryable(Exception):
        status_code = 429
    pauses = []
    ratings = json.dumps({f"s{i}": 2 for i in range(1, 17)})
    client = FakeClient([Retryable("sensitive server body"), ratings])
    task = runner.Task("person-a", "qwen-8b", "name-only", "neutral", 0)
    runner.obtain(client, task, "invented", "vllm", "https://example.invalid/v1", tmp_path / "one", sleep=pauses.append)
    assert pauses == [1] and len(client.calls) == 2
    with pytest.raises(RuntimeError) as error:
        runner.obtain(FakeClient([ValueError("secret credential")]), task, "invented", "vllm", "https://example.invalid/v1", tmp_path / "two")
    assert "secret credential" not in str(error.value)


def test_modal_serving_uses_argv_and_requires_authentication():
    source = ast.parse((ROOT / "scripts" / "modal_app.py").read_text())
    assignment = next(n for n in source.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MODELS" for t in n.targets))
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "server_command")
    namespace = {"PORT": 8000}
    exec(compile(ast.Module(body=[assignment, function], type_ignores=[]), "modal-config", "exec"), namespace)
    with pytest.raises(ValueError, match="required"):
        namespace["server_command"]("qwen-8b", "")
    key = "invented value; shell substitution is not executed"
    argv = namespace["server_command"]("qwen-8b", key)
    assert argv[-2:] == ["--api-key", key]
    assert argv[argv.index("--revision") + 1] == "b968826d9c46dd6066d109eabc6255188de91218"
    popen = [n for n in ast.walk(source) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "Popen"]
    assert len(popen) == 1
    assert not any(k.arg == "shell" for k in popen[0].keywords)


def test_identical_speech_slots_keep_independent_requests():
    a = runner.Task("person-a", "qwen-8b", "named-speech", "neutral", 0, "a.txt")
    b = runner.Task("person-a", "qwen-8b", "named-speech", "neutral", 0, "b.txt")
    options = runner.request_options(a, "same invented speech", "vllm")
    assert runner.request_hash(options, "https://example.invalid/v1", a) != runner.request_hash(options, "https://example.invalid/v1", b)
