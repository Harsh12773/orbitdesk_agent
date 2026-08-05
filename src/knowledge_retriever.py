import sys
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import KNOWLEDGE_BASE_DIR, RESOLVED_CASES_PATH, EMBEDDING_MODEL_NAME

class KnowledgeRetriever:
    """RAG Search Engine using Hugging Face SentenceTransformer Embeddings (PDF Req #2)."""
    def __init__(self):
        self.documents = []
        self.load_documents()
        self.load_resolved_cases()
        self.model = None
        self.doc_embeddings = None
        self.init_embedding_model()

    def load_documents(self):
        """Loads and indexes all 10 Markdown KB documents."""
        if not KNOWLEDGE_BASE_DIR.exists():
            return

        for file_path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
            filename = file_path.name
            doc_id = "KB-" + filename.split("_")[0].zfill(3)
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            clean_content = re.sub(r'^---.*?---\s*', '', content, flags=re.DOTALL).strip()

            self.documents.append({
                "source_id": doc_id,
                "title": filename,
                "passage": clean_content,
                "is_superseded": False
            })

    def load_resolved_cases(self):
        """Loads resolved support cases, excluding superseded cases from active guidance."""
        if not RESOLVED_CASES_PATH.exists():
            return

        with open(RESOLVED_CASES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        for case in data.get("cases", []):
            case_id = case.get("case_id")
            title = case.get("title", "")
            symptoms = " ".join(case.get("symptoms", []))
            resolution = " ".join(case.get("resolution", []))
            is_superseded = (case.get("status") == "superseded")

            passage = f"Case Title: {title}\nSymptoms: {symptoms}\nResolution: {resolution}"
            
            self.documents.append({
                "source_id": case_id,
                "title": title,
                "passage": passage,
                "is_superseded": is_superseded
            })

    def init_embedding_model(self):
        """Loads local Hugging Face embedding model for dense vector search."""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[KnowledgeRetriever] Loading local Hugging Face embedding model: {EMBEDDING_MODEL_NAME}...")
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            passages = [doc["passage"] for doc in self.documents]
            self.doc_embeddings = self.model.encode(passages, convert_to_tensor=True)
            print(f"[KnowledgeRetriever] Embeddings generated for {len(passages)} passages.")
        except Exception as e:
            print(f"[KnowledgeRetriever] Embedding load note ({e}). Using lexical fallback.")
            self.model = None

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Searches documents using Hugging Face SentenceTransformer vector similarity."""
        if not self.documents:
            return []

        # Exclude superseded cases from current guidance (PDF Rule)
        candidate_indices = [
            i for i, doc in enumerate(self.documents)
            if not doc["is_superseded"]
        ]

        if self.model is not None and self.doc_embeddings is not None:
            from sentence_transformers import util
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            candidate_embeddings = self.doc_embeddings[candidate_indices]
            scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
            
            top_results = []
            for idx, score_val in enumerate(scores):
                doc_idx = candidate_indices[idx]
                doc = self.documents[doc_idx].copy()
                doc["score"] = float(score_val)
                top_results.append(doc)
            
            top_results.sort(key=lambda x: x["score"], reverse=True)
            return top_results[:top_k]
        else:
            # Fallback search
            q_lower = query.lower()
            query_words = set(re.findall(r'\w+', q_lower))
            scored_docs = []
            for idx in candidate_indices:
                doc = self.documents[idx].copy()
                doc_words = set(re.findall(r'\w+', doc["passage"].lower()))
                overlap = len(query_words.intersection(doc_words))
                doc["score"] = overlap
                scored_docs.append(doc)
            
            scored_docs.sort(key=lambda x: x["score"], reverse=True)
            return scored_docs[:top_k]