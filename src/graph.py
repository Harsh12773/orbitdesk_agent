import sys
from pathlib import Path
from typing import Dict, Any, Literal

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from state import AgentState
from knowledge_retriever import KnowledgeRetriever
from local_llm import LocalLLM
from verifier import Verifier
from config import MAX_RETRIES

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    END = "__END__"

class SupportAgentGraph:
    """LangGraph State Machine for OrbitDesk Support Agent Network."""
    def __init__(self):
        self.retriever = KnowledgeRetriever()
        self.llm = LocalLLM()
        self.verifier = Verifier(llm=self.llm)
        self.graph = self._build_graph() if HAS_LANGGRAPH else None

    def _build_graph(self):
        """Constructs LangGraph state graph workflow with conditional edges and retries."""
        workflow = StateGraph(AgentState)

        workflow.add_node("triage", self.triage_node)
        workflow.add_node("retrieve", self.retrieval_node)
        workflow.add_node("generate", self.generation_node)
        workflow.add_node("verify", self.verification_node)
        workflow.add_node("clarification", self.clarification_node)
        workflow.add_node("escalation", self.escalation_node)
        workflow.add_node("out_of_scope", self.out_of_scope_node)
        workflow.add_node("safe_failure", self.safe_failure_node)

        workflow.set_entry_point("triage")

        workflow.add_edge("triage", "retrieve")

        workflow.add_conditional_edges(
            "retrieve",
            self.route_after_triage,
            {
                "generate": "generate",
                "clarification": "clarification",
                "escalation": "escalation",
                "out_of_scope": "out_of_scope"
            }
        )

        workflow.add_edge("generate", "verify")

        workflow.add_conditional_edges(
            "verify",
            self.route_after_verification,
            {
                "end": END,
                "retry": "generate",
                "safe_failure": "safe_failure"
            }
        )

        workflow.add_edge("clarification", END)
        workflow.add_edge("escalation", END)
        workflow.add_edge("out_of_scope", END)
        workflow.add_edge("safe_failure", END)

        return workflow.compile()

    # --- NODE IMPLEMENTATIONS ---

    def triage_node(self, state: AgentState) -> Dict[str, Any]:
        q = state["question"]
        trace = state.get("node_trace", []) + ["triage"]

        classification = "answerable" # Default fallback
        reason = "Fallback classification"

        if self.llm and self.llm.pipeline is not None:
            prompt = (
                "<|im_start|>system\nClassify the following user request into exactly one of these categories: "
                "'answerable', 'requires_clarification', 'requires_escalation', 'out_of_scope'. Output ONLY the category name. "
                "If a request asks for refunds, compensation, or legal advice, classify as 'out_of_scope'. "
                "If a request is too vague and lacks diagnostic details, classify as 'requires_clarification'.\n<|im_end|>\n"
                f"<|im_start|>user\nRequest: {q}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = self.llm.pipeline(prompt, max_new_tokens=10, return_full_text=False)
                    gen_text = out[0]["generated_text"].strip().lower()
                    
                    if "out_of_scope" in gen_text or "out of scope" in gen_text:
                        classification = "out_of_scope"
                    elif "escalation" in gen_text:
                        classification = "requires_escalation"
                    elif "clarification" in gen_text:
                        classification = "requires_clarification"
                    else:
                        classification = "answerable"
                    reason = "Dynamically classified by LLM"
            except Exception:
                pass

        return {"classification": classification, "reason": reason, "node_trace": trace}

    def retrieval_node(self, state: AgentState) -> Dict[str, Any]:
        trace = state.get("node_trace", []) + ["retrieve"]
        docs = self.retriever.search(state["question"], top_k=3)
        return {"retrieved_docs": docs, "node_trace": trace}

    def generation_node(self, state: AgentState) -> Dict[str, Any]:
        trace = state.get("node_trace", []) + ["generate"]
        output = self.llm.generate_response(state["question"], state.get("retrieved_docs", []), state.get("classification"))
        return {
            "generated_answer": output["answer"],
            "sources": output["sources"],
            "confidence": output["confidence"],
            "requires_human": output["requires_human"],
            "reason": output["reason"],
            "node_trace": trace
        }

    def verification_node(self, state: AgentState) -> Dict[str, Any]:
        trace = state.get("node_trace", []) + ["verify"]
        response_dict = {
            "classification": state.get("classification"),
            "answer": state.get("generated_answer"),
            "sources": state.get("sources"),
            "confidence": state.get("confidence"),
            "requires_human": state.get("requires_human"),
            "reason": state.get("reason")
        }
        passed, warnings = self.verifier.verify(response_dict, state.get("retrieved_docs", []))
        
        current_retry = state.get("retry_count", 0)
        if not passed:
            current_retry += 1

        return {
            "verification_passed": passed,
            "retry_count": current_retry,
            "warnings": warnings,
            "node_trace": trace
        }

    def clarification_node(self, state: AgentState) -> Dict[str, Any]:
        """Dynamically generates clarification response using LLM."""
        trace = state.get("node_trace", []) + ["clarification"]
        output = self.llm.generate_response(state["question"], state.get("retrieved_docs", []), "requires_clarification")
        return {
            "classification": "requires_clarification",
            "generated_answer": output["answer"],
            "sources": output["sources"],
            "confidence": output["confidence"],
            "requires_human": output["requires_human"],
            "reason": output["reason"],
            "node_trace": trace
        }

    def escalation_node(self, state: AgentState) -> Dict[str, Any]:
        """Dynamically generates escalation response using LLM."""
        trace = state.get("node_trace", []) + ["escalation"]
        output = self.llm.generate_response(state["question"], state.get("retrieved_docs", []), "requires_escalation")
        return {
            "classification": "requires_escalation",
            "generated_answer": output["answer"],
            "sources": output["sources"],
            "confidence": output["confidence"],
            "requires_human": output["requires_human"],
            "reason": output["reason"],
            "node_trace": trace
        }

    def out_of_scope_node(self, state: AgentState) -> Dict[str, Any]:
        """Dynamically generates out of scope response using LLM."""
        trace = state.get("node_trace", []) + ["out_of_scope"]
        output = self.llm.generate_response(state["question"], state.get("retrieved_docs", []), "out_of_scope")
        return {
            "classification": "out_of_scope",
            "generated_answer": output["answer"],
            "sources": output["sources"],
            "confidence": output["confidence"],
            "requires_human": output["requires_human"],
            "reason": output["reason"],
            "node_trace": trace
        }

    def safe_failure_node(self, state: AgentState) -> Dict[str, Any]:
        """Dynamically generates safe failure response using LLM."""
        trace = state.get("node_trace", []) + ["safe_failure"]
        output = self.llm.generate_response(state["question"], [], "safe_failure")
        return {
            "classification": "safe_failure",
            "generated_answer": output["answer"],
            "sources": [],
            "confidence": 0.0,
            "requires_human": True,
            "reason": "Verification failed after retries",
            "node_trace": trace
        }

    # --- ROUTING LOGIC ---

    def route_after_triage(self, state: AgentState) -> Literal["generate", "clarification", "escalation", "out_of_scope"]:
        cls = state.get("classification")
        if cls == "requires_clarification":
            return "clarification"
        elif cls == "requires_escalation":
            return "escalation"
        elif cls == "out_of_scope":
            return "out_of_scope"
        return "generate"

    def route_after_verification(self, state: AgentState) -> Literal["end", "retry", "safe_failure"]:
        if state.get("verification_passed", False):
            return "end"
        if state.get("retry_count", 0) <= MAX_RETRIES:
            return "retry"
        return "safe_failure"

    def run(self, question: str) -> Dict[str, Any]:
        """Executes graph workflow and returns structured JSON."""
        state = {
            "question": question,
            "classification": "answerable",
            "retrieved_docs": [],
            "generated_answer": "",
            "sources": [],
            "confidence": 0.0,
            "requires_human": False,
            "reason": "",
            "retry_count": 0,
            "verification_passed": False,
            "warnings": [],
            "node_trace": []
        }

        if self.graph is not None:
            final_state = self.graph.invoke(state)
        else:
            final_state = state.copy()
            final_state.update(self.triage_node(final_state))
            final_state.update(self.retrieval_node(final_state))
            route = self.route_after_triage(final_state)
            
            if route == "clarification":
                final_state.update(self.clarification_node(final_state))
            elif route == "escalation":
                final_state.update(self.escalation_node(final_state))
            elif route == "out_of_scope":
                final_state.update(self.out_of_scope_node(final_state))
            else:
                final_state.update(self.generation_node(final_state))
                final_state.update(self.verification_node(final_state))
                
                verif_route = self.route_after_verification(final_state)
                if verif_route == "safe_failure":
                    final_state.update(self.safe_failure_node(final_state))

        return {
            "classification": final_state.get("classification"),
            "answer": final_state.get("generated_answer"),
            "sources": final_state.get("sources", []),
            "confidence": final_state.get("confidence", 0.90),
            "requires_human": final_state.get("requires_human", False),
            "reason": final_state.get("reason", ""),
            "node_trace": final_state.get("node_trace", [])
        }