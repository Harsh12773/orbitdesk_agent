from typing import List, Dict, Any, Optional, TypedDict

# Shared notebook passed between flowchart nodes
class AgentState(TypedDict):
    question_id: Optional[str]
    question: str
    classification: str          # "answerable", "requires_clarification", "requires_escalation", "out_of_scope"
    retrieved_docs: List[Dict[str, Any]]
    generated_answer: str
    sources: List[Dict[str, str]]
    confidence: float
    requires_human: bool
    reason: str
    clarification_question: Optional[str]
    warnings: List[str]
    retry_count: int
    verification_passed: bool
    node_trace: List[str]