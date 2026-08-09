import sys
from pathlib import Path
from types import ModuleType

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

triage_mock = ModuleType("triage")
triage_mock.classify_question = lambda question: "answerable"

rag_mock = ModuleType("rag")
rag_mock.generate_answer = lambda question, revision_note=None: ("", [])

verifier_mock = ModuleType("verifier")
verifier_mock.verify_answer = lambda question, answer, results: (True, "")

formatter_mock = ModuleType("output_formatter")
formatter_mock.format_output = lambda state: state

sys.modules["triage"] = triage_mock
sys.modules["rag"] = rag_mock
sys.modules["verifier"] = verifier_mock
sys.modules["output_formatter"] = formatter_mock

from graph import route_after_triage


def test_out_of_scope_routing():
    state = {
        "classification": "out_of_scope"
    }

    route = route_after_triage(state)

    assert route == "out_of_scope"