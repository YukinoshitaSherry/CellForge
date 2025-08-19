#!/usr/bin/env python3
"""
Enhanced Plan Storage Module
Integrates JSON file storage with vector database for knowledge support
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPlanStorage:
    """Enhanced plan storage with vector database integration"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent  # scAgents root
        self.storage_dir = self.project_root / "cellforge" / "data" / "plans"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize vector database connection
        self._init_vector_db()
    
    def _init_vector_db(self):
        """Initialize vector database connection"""
        try:
            import json as json_module
            config_path = self.project_root / "config.json"
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json_module.load(f)
                
                from qdrant_client import QdrantClient
                self.qdrant_tmp = QdrantClient(
                    url=config['qdrant_config']['cellforge_tmp']['url'],
                    api_key=config['qdrant_config']['cellforge_tmp']['api_key']
                )
                logger.info("Successfully connected to cellforge_tmp vector database")
            else:
                logger.warning("Config file not found, vector database disabled")
                self.qdrant_tmp = None
                
        except Exception as e:
            logger.warning(f"Could not initialize vector database: {e}")
            self.qdrant_tmp = None
    
    def save_research_plan(self, 
                          plan_data: Dict[str, Any], 
                          query: str,
                          retrieved_documents: List[Dict[str, Any]] = None) -> str:
        """
        Save research plan to JSON file and vector database
        
        Args:
            plan_data: The generated research plan
            query: Original query
            retrieved_documents: Retrieved documents for context
            
        Returns:
            File path of saved JSON
        """
        try:
            # Create enhanced plan structure
            enhanced_plan = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "plan_id": str(uuid.uuid4()),
                    "query": query,
                    "version": "1.0"
                },
                "research_plan": plan_data,
                "retrieved_documents": retrieved_documents or [],
                "vector_db_id": None
            }
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_plan_{timestamp}.json"
            file_path = self.storage_dir / filename
            
            # Save to JSON file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_plan, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Research plan saved to: {file_path}")
            
            # Store in vector database if available
            if self.qdrant_tmp:
                vector_id = self._store_in_vector_db(enhanced_plan, file_path)
                enhanced_plan["vector_db_id"] = vector_id
                
                # Update JSON file with vector ID
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_plan, f, indent=2, ensure_ascii=False)
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving research plan: {e}")
            raise
    
    def _store_in_vector_db(self, plan_data: Dict[str, Any], file_path: Path) -> str:
        """Store plan data in vector database"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Initialize embedding model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create text content for embedding
            content_parts = []
            
            # Add query
            content_parts.append(f"Query: {plan_data['metadata']['query']}")
            
            # Add research plan content
            plan_content = plan_data['research_plan']
            if isinstance(plan_content, dict):
                for key, value in plan_content.items():
                    if isinstance(value, list):
                        content_parts.append(f"{key}: {', '.join(map(str, value))}")
                    else:
                        content_parts.append(f"{key}: {value}")
            
            # Add retrieved documents
            for doc in plan_data.get('retrieved_documents', []):
                content_parts.append(f"Document: {doc.get('title', '')} - {doc.get('content', '')[:200]}")
            
            # Combine all content
            full_content = " | ".join(content_parts)
            
            # Generate embedding
            embedding = model.encode(full_content).tolist()
            
            # Create unique ID
            vector_id = str(uuid.uuid4())
            
            # Store in vector database
            self.qdrant_tmp.upsert(
                collection_name="research_plans",
                points=[{
                    "id": vector_id,
                    "vector": embedding,
                    "payload": {
                        "content": full_content,
                        "file_path": str(file_path),
                        "plan_id": plan_data['metadata']['plan_id'],
                        "query": plan_data['metadata']['query'],
                        "timestamp": plan_data['metadata']['generated_at']
                    }
                }]
            )
            
            logger.info(f"Plan stored in vector database with ID: {vector_id}")
            return vector_id
            
        except Exception as e:
            logger.error(f"Error storing in vector database: {e}")
            return None
    
    def get_plan_file_path(self, plan_id: str = None, latest: bool = True) -> Optional[str]:
        """
        Get research plan file path
        
        Args:
            plan_id: Specific plan ID to find
            latest: If True and no plan_id, return latest plan
            
        Returns:
            File path or None if not found
        """
        try:
            if plan_id:
                # Search for specific plan
                for file_path in self.storage_dir.glob("research_plan_*.json"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if data.get('metadata', {}).get('plan_id') == plan_id:
                                return str(file_path)
                    except:
                        continue
            elif latest:
                # Get latest plan
                files = list(self.storage_dir.glob("research_plan_*.json"))
                if files:
                    latest_file = max(files, key=lambda x: x.stat().st_mtime)
                    return str(latest_file)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting plan file path: {e}")
            return None
    
    def load_research_plan(self, file_path: str = None, plan_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Load research plan from file
        
        Args:
            file_path: Direct file path
            plan_id: Plan ID to search for
            
        Returns:
            Plan data or None if not found
        """
        try:
            if file_path:
                target_path = Path(file_path)
            else:
                target_path = Path(self.get_plan_file_path(plan_id=plan_id))
            
            if target_path and target_path.exists():
                with open(target_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading research plan: {e}")
            return None
    
    def search_similar_plans(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar research plans in vector database
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of similar plans
        """
        try:
            if not self.qdrant_tmp:
                logger.warning("Vector database not available for search")
                return []
            
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Generate query embedding
            query_embedding = model.encode(query).tolist()
            
            # Search in vector database
            results = self.qdrant_tmp.search(
                collection_name="research_plans",
                query_vector=query_embedding,
                limit=top_k
            )
            
            # Load full plan data for each result
            similar_plans = []
            for result in results:
                file_path = result.payload.get('file_path')
                if file_path and Path(file_path).exists():
                    plan_data = self.load_research_plan(file_path)
                    if plan_data:
                        similar_plans.append({
                            'plan_data': plan_data,
                            'similarity_score': result.score,
                            'file_path': file_path
                        })
            
            return similar_plans
            
        except Exception as e:
            logger.error(f"Error searching similar plans: {e}")
            return []

# Global instance
plan_storage = EnhancedPlanStorage() 