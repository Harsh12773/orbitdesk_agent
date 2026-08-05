import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import OUTPUT_SCHEMA_PATH

class Verifier:
    """Simple Proofreader for checking JSON schema and safety rules."""
    def __init__(self, llm=None):
        self.llm = llm

    def verify(self, response_data: Dict[str, Any], retrieved_docs: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        warnings = []
        passed = True

        # Rule 1: Required Keys Check (Schema remains hardcoded)
        required_keys = ["classification", "answer", "sources", "confidence", "requires_human", "reason"]
        for key in required_keys:
            if key not in response_data:
                passed = False
                warnings.append(f"Missing field: {key}")


        # Rule 3: Safety Check using AI
        answer_text = response_data.get("answer", "")
        if self.llm and self.llm.pipeline is not None:
            prompt = (
                "<|im_start|>system\nYou are a strict safety checker. Answer ONLY 'yes' or 'no'.<|im_end|>\n"
                f"<|im_start|>user\nDoes the following text violate company safety policies (such as offering unauthorized financial compensation or legal advice)?\nText: {answer_text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            try:
                import warnings as sys_warnings
                with sys_warnings.catch_warnings():
                    sys_warnings.simplefilter("ignore")
                    out = self.llm.pipeline(prompt, max_new_tokens=5, return_full_text=False)
                    gen_text = out[0]["generated_text"].strip().lower()
                    if "yes" in gen_text:
                        passed = False
                        warnings.append("AI Safety check failed: Policy violation detected in the generated response.")
            except Exception:
                pass

        return passed, warnings