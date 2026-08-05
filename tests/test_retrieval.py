import sys
import json
from pathlib import Path

# Add src directory directly to Python's search path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytest
from knowledge_retriever import KnowledgeRetriever

def load_test_data():
    data_path = Path(__file__).resolve().parent / "test_data.json"
    with open(data_path, "r") as f:
        return json.load(f)["retrieval_tests"]

@pytest.fixture
def retriever():
    return KnowledgeRetriever()

def test_dynamic_retrieval(retriever):
    test_cases = load_test_data()
    
    # Test case 1: precedence and superseded filtering
    tc1 = test_cases[0]
    results1 = retriever.search(tc1["query"], top_k=tc1["top_k"])
    source_ids1 = [r["source_id"] for r in results1]
    assert tc1["excluded_source"] not in source_ids1

    # Test case 2: timezone document retrieval
    tc2 = test_cases[1]
    results2 = retriever.search(tc2["query"], top_k=tc2["top_k"])
    assert len(results2) > 0
    top_doc = results2[0]
    assert top_doc["source_id"] in tc2["expected_any_of"]