"""
Model Recommender Service - RAG-based model selection engine.
Uses semantic search + LLM ranking to recommend the best model for a user's intent.
"""
import os
import sys
import sqlite3
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))
from services.llm_service import query_llm_gemini
from config.param_mappings import get_params_from_architectures
import json

# Paths
from core.config import DATA_DIR, DB_PATH
EMBEDDINGS_FILE = DATA_DIR / "model_embeddings.pkl"

class ModelRecommender:
    """
    RAG-based model recommendation engine.
    
    Pipeline:
    1. Hard Filter: Filter by task type
    2. Semantic Search: Find top-K similar models using embeddings
    3. LLM Ranking: Use Gemini to select the best model from candidates
    """
    
    def __init__(self):
        """Initialize the recommender by loading the index and embedding model."""
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index_data = None
        self.db_conn = None
        self._load_index()
        self._connect_db()
    
    def _load_index(self):
        """Load the pre-built vector index."""
        if not EMBEDDINGS_FILE.exists():
            raise FileNotFoundError(
                f"Embeddings file not found at {EMBEDDINGS_FILE}. "
                "Please run scripts/build_index.py first."
            )
        
        with open(EMBEDDINGS_FILE, 'rb') as f:
            self.index_data = pickle.load(f)
    
    def _connect_db(self):
        """Connect to the model registry database."""
        if not DB_PATH.exists():
            raise FileNotFoundError(f"Model registry database not found at {DB_PATH}")
        
        self.db_conn = sqlite3.connect(DB_PATH)
    
    def _get_model_details(self, model_id: str) -> Optional[Dict]:
        """Fetch full model details from the database."""
        c = self.db_conn.cursor()
        c.execute("""
            SELECT id, task, downloads, likes, architectures, 
                   max_position_embeddings, vocab_size, model_type
            FROM models WHERE id = ?
        """, (model_id,))
        
        row = c.fetchone()
        if not row:
            return None
        
        return {
            'id': row[0],
            'task': row[1],
            'downloads': row[2],
            'likes': row[3],
            'architectures': json.loads(row[4]) if row[4] else [],
            'max_position_embeddings': row[5],
            'vocab_size': row[6],
            'model_type': row[7]
        }
    
    def recommend(
        self, 
        user_intent: str, 
        task_type: Optional[str] = None,
        top_k: int = 5
    ) -> Dict:
        """
        Recommend the best model for a given user intent.
        
        Args:
            user_intent: Natural language description of the task
            task_type: Optional task filter (e.g., 'text-classification')
            top_k: Number of candidates to retrieve for LLM ranking
        
        Returns:
            Dictionary with:
                - selected_model_id: The recommended model
                - reasoning: Explanation from the LLM
                - suggested_hyperparams: Parameter recommendations
                - candidates: List of all candidates considered
        """
        # Stage 1: Hard Filter by task type
        if task_type:
            filtered_indices = self._filter_by_task(task_type)
        else:
            filtered_indices = list(range(len(self.index_data['model_ids'])))
        
        if not filtered_indices:
            return {
                "error": f"No models found for task type: {task_type}",
                "selected_model_id": None,
                "reasoning": None,
                "suggested_hyperparams": {}
            }
        
        # Stage 2: Semantic Search
        candidates = self._semantic_search(user_intent, filtered_indices, top_k)
        
        # Stage 3: LLM Ranking
        recommendation = self._llm_rank(user_intent, candidates)
        
        return recommendation
    
    def _filter_by_task(self, task_type: str) -> List[int]:
        """Filter model indices by task type."""
        # Normalize task type to match DB format
        # Frontend may send "Text Classification", DB has "text-classification"
        if task_type:
            task_type = task_type.lower().replace(" ", "-")
        
        filtered_indices = []
        
        for idx, model_id in enumerate(self.index_data['model_ids']):
            model = self._get_model_details(model_id)
            if model and model['task'] == task_type:
                filtered_indices.append(idx)
        
        return filtered_indices
    
    def _semantic_search(
        self, 
        query: str, 
        candidate_indices: List[int], 
        top_k: int
    ) -> List[Dict]:
        """
        Perform semantic search using cosine similarity.
        
        Returns:
            List of candidate models with their similarity scores
        """
        # Encode the query
        query_embedding = self.embedder.encode([query])
        
        # Get embeddings for candidates
        candidate_embeddings = self.index_data['embeddings'][candidate_indices]
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        # Get top-K indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Map back to global indices
        global_indices = [candidate_indices[i] for i in top_indices]
        
        # Fetch full model details
        candidates = []
        for idx, global_idx in enumerate(global_indices):
            model_id = self.index_data['model_ids'][global_idx]
            model_details = self._get_model_details(model_id)
            
            if model_details:
                model_details['similarity_score'] = float(similarities[top_indices[idx]])
                candidates.append(model_details)
        
        return candidates
    
    def _llm_rank(self, user_intent: str, candidates: List[Dict]) -> Dict:
        """
        Use Gemini to rank candidates and provide reasoning for each.
        """
        # Construct prompt
        candidates_summary = []
        for i, candidate in enumerate(candidates, 1):
            candidates_summary.append(f"""
{i}. Model: {candidate['id']}
   - Task: {candidate['task']}
   - Architecture: {candidate.get('architectures', ['Unknown'])[0] if candidate.get('architectures') else 'Unknown'}
   - Popularity: {candidate['downloads']:,} downloads, {candidate['likes']} likes
   - Similarity Score: {candidate['similarity_score']:.3f}
""")
        
        prompt = f"""
You are an ML model selection expert. Analyze the user's intent and evaluate ALL {len(candidates)} candidates.

User Intent: "{user_intent}"

Candidates:
{"".join(candidates_summary)}

Return ONLY valid JSON with this structure:
{{
  "recommended_id": "exact model ID you recommend as THE BEST",
  "candidates": [
    {{
      "id": "exact model ID from the list",
      "reasoning": "1-2 sentence explanation of why this model fits the user intent",
      "suitability_score": 0.0-1.0,
      "suggested_hyperparams": {{
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "batch_size": 8
      }}
    }},
    ... (repeat for ALL {len(candidates)} candidates)
  ]
}}

IMPORTANT:
1. Provide reasoning for EVERY candidate, not just the recommended one
2. The "recommended_id" should match one of the candidate IDs
3. Order candidates by your suitability_score (best first)
4. Consider: task alignment, popularity, architecture suitability, semantic similarity
""".strip()
        
        try:
            response = query_llm_gemini(prompt)
            
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                llm_result = json.loads(json_match.group(0))
                
                # Enrich each candidate with full model details and architecture-specific params
                enriched_candidates = []
                for llm_candidate in llm_result.get('candidates', []):
                    model_id = llm_candidate['id']
                    
                    # Find the original candidate with full details
                    original_candidate = next(
                        (c for c in candidates if c['id'] == model_id),
                        None
                    )
                    
                    if original_candidate:
                        # Get architecture-specific parameters
                        arch_params = {}
                        if original_candidate.get('architectures'):
                            arch_params = get_params_from_architectures(original_candidate['architectures'])
                        
                        enriched_candidates.append({
                            'id': model_id,
                            'task': original_candidate['task'],
                            'architectures': original_candidate.get('architectures', []),
                            'downloads': original_candidate['downloads'],
                            'likes': original_candidate['likes'],
                            'similarity_score': original_candidate['similarity_score'],
                            'reasoning': llm_candidate.get('reasoning', 'Good fit for this task'),
                            'suitability_score': llm_candidate.get('suitability_score', 0.8),
                            'suggested_hyperparams': llm_candidate.get('suggested_hyperparams', {}),
                            'all_params': arch_params  # Full parameter schema for this model
                        })
                
                return {
                    'recommended_id': llm_result.get('recommended_id', enriched_candidates[0]['id'] if enriched_candidates else None),
                    'candidates': enriched_candidates,
                    'selected_model_id': llm_result.get('recommended_id'),  # For backward compatibility
                    'reasoning': next((c['reasoning'] for c in enriched_candidates if c['id'] == llm_result.get('recommended_id')), None)
                }
            else:
                # Fallback: return all candidates with basic reasoning
                return self._fallback_recommendation(candidates)
                
        except Exception as e:
            print(f"LLM ranking failed: {e}")
            return self._fallback_recommendation(candidates)
    
    def _fallback_recommendation(self, candidates: List[Dict]) -> Dict:
        """Fallback recommendation if LLM fails - return all candidates."""
        if not candidates:
            return {
                "error": "No candidates available",
                "recommended_id": None,
                "candidates": []
            }
        
        # Enrich all candidates
        enriched_candidates = []
        for candidate in candidates:
            # Get architecture-specific parameters
            arch_params = {}
            if candidate.get('architectures'):
                arch_params = get_params_from_architectures(candidate['architectures'])
            
            enriched_candidates.append({
                'id': candidate['id'],
                'task': candidate['task'],
                'architectures': candidate.get('architectures', []),
                'downloads': candidate['downloads'],
                'likes': candidate['likes'],
                'similarity_score': candidate['similarity_score'],
                'reasoning': f"High semantic similarity ({candidate['similarity_score']:.3f}) and popularity ({candidate['downloads']:,} downloads)",
                'suitability_score': candidate['similarity_score'],
                'suggested_hyperparams': {
                    'learning_rate': 2e-5,
                    'num_epochs': 3,
                    'batch_size': 8
                },
                'all_params': arch_params
            })
        
        # Sort by similarity score
        enriched_candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return {
            'recommended_id': enriched_candidates[0]['id'],
            'candidates': enriched_candidates,
            'selected_model_id': enriched_candidates[0]['id'],  # For backward compatibility
            'reasoning': enriched_candidates[0]['reasoning']
        }
    
    def __del__(self):
        """Cleanup database connection."""
        if self.db_conn:
            self.db_conn.close()
