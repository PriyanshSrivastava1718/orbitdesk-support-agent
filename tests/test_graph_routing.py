import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from graph import route_after_triage


def test_out_of_scope_routing():
    state = {
        "classification": "out_of_scope"
    }

    route = route_after_triage(state)

    assert route == "out_of_scope"