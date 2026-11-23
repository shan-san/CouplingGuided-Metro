import json
import os
import uuid
from typing import List, Dict, Any

class RAGManager:
    """
    Manages the storage and retrieval of historical cases for the MetroRL RAG system.
    """
    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self.cases = []
        if db_path and os.path.exists(db_path):
            self.load_database(db_path)

    def add_case(self, scenario_desc: str, solution_plan: str, metadata: Dict[str, Any] = None):
        """
        Adds a new case to the in-memory list.
        """
        case = {
            "id": str(uuid.uuid4())[:8], # Short ID
            "scenario": scenario_desc,
            "solution": solution_plan,
            "metadata": metadata or {}
        }
        self.cases.append(case)

    def save_database(self, path: str = None):
        """
        Saves the current cases to a JSONL file.
        """
        target_path = path or self.db_path
        if not target_path:
            raise ValueError("No path specified for saving RAG database.")
        
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, 'w', encoding='utf-8') as f:
            for case in self.cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")

    def load_database(self, path: str):
        """
        Loads cases from a JSONL file.
        """
        self.cases = []
        if not os.path.exists(path):
            return
            
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.cases.append(json.loads(line))

    def retrieve(self, current_scenario: str, k: int = 3) -> List[Dict]:
        """
        Retrieves the top-k most similar cases.
        
        Note: This is a basic implementation using Jaccard similarity on tokens.
        For production, replace this with vector embeddings (e.g., OpenAI, SentenceTransformers)
        and a vector database (e.g., FAISS, ChromaDB).
        """
        if not self.cases:
            return []

        # Simple token-based similarity
        def tokenize(text):
            return set(text.lower().replace('.', ' ').replace(',', ' ').split())

        query_tokens = tokenize(current_scenario)
        
        scored_cases = []
        for case in self.cases:
            case_tokens = tokenize(case["scenario"])
            if not case_tokens:
                score = 0
            else:
                intersection = query_tokens.intersection(case_tokens)
                union = query_tokens.union(case_tokens)
                score = len(intersection) / len(union)
            
            # Boost score if disruptions match exactly (heuristic)
            if "disrupted" in current_scenario and "disrupted" in case["scenario"]:
                 # Extract disruption IDs if possible, simplified here
                 pass

            scored_cases.append((score, case))
        
        # Sort by score descending
        scored_cases.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k
        return [c[1] for c in scored_cases[:k]]
