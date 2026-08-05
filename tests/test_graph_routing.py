import sys
import json
from pathlib import Path

# Add src directory directly to Python's search path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytest
from graph import SupportAgentGraph

def load_test_data():
    data_path = Path(__file__).resolve().parent / "test_data.json"
    with open(data_path, "r") as f:
        return json.load(f)["routing_tests"]

@pytest.fixture
def agent_graph():
    return SupportAgentGraph()

@pytest.mark.parametrize("test_case", load_test_data())
def test_dynamic_routing(agent_graph, monkeypatch, test_case):
    q = test_case["question"]
    expected_class = test_case["expected_classification"]
    expected_nodes = test_case["expected_nodes"]

    # If this is the verification failure case, mock the LLM to trigger a safety violation
    if expected_class == "safe_failure":
        def mock_synthesize(*args, **kwargs):
            return "I will provide unauthorized financial compensation to your account immediately.", "NONE"
        monkeypatch.setattr(agent_graph.llm, "_synthesize_answer", mock_synthesize)

    res = agent_graph.run(q)
    
    assert res["classification"] == expected_class
    
    for node in expected_nodes:
        assert node in res["node_trace"]
        
    if expected_class == "requires_clarification":
        assert res["answer"] is not None
    elif expected_class in ["requires_escalation", "safe_failure"]:
        assert res["requires_human"] is True
    elif expected_class == "out_of_scope":
        assert res["requires_human"] is False