"""
MCP-compatible Knowledge Base for Task Analysis
Stores and retrieves knowledge from dual Qdrant databases without repeated searches
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from sentence_transformers import SentenceTransformer
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("⚠️  Qdrant client not available. Knowledge base will use local storage.")

@dataclass
class KnowledgeItem:
    """Knowledge item with metadata"""
    content: Dict[str, Any]
    source: str
    relevance_score: float
    timestamp: datetime
    metadata: Dict[str, Any]
    knowledge_type: str  # 'papers', 'experimental_designs', 'implementation_guides', etc.
    database: str  # 'CellForge' or 'cellforge_tmp'

class MCPKnowledgeBase:
    """
    MCP-compatible Knowledge Base that stores and retrieves knowledge
    from dual Qdrant databases without repeated searches. Compatible with various LLM models.
    """
    
    def __init__(self):
        self.knowledge_store: Dict[str, List[KnowledgeItem]] = {}
        self.encoder = None
        self.qdrant_cellforge = None  # 主数据库
        self.qdrant_tmp = None        # 临时数据库
        
        # Load configuration
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            # Initialize encoder
            if QDRANT_AVAILABLE:
                try:
                    self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                    print("✅ Sentence transformer encoder initialized")
                except Exception as e:
                    print(f"⚠️  Failed to initialize encoder: {e}")
            
            # Initialize dual Qdrant clients
            if QDRANT_AVAILABLE:
                try:
                    # 主数据库 - CellForge
                    self.qdrant_cellforge = QdrantClient(
                        url=config['qdrant_config']['CelloFrge']['url'],
                        api_key=config['qdrant_config']['CelloFrge']['api_key']
                    )
                    print("✅ CellForge Qdrant client initialized")
                    
                    # 临时数据库 - cellforge_tmp
                    self.qdrant_tmp = QdrantClient(
                        url=config['qdrant_config']['cellforge_tmp']['url'],
                        api_key=config['qdrant_config']['cellforge_tmp']['api_key']
                    )
                    print("✅ cellforge_tmp Qdrant client initialized")
                    
                except Exception as e:
                    print(f"⚠️  Failed to initialize Qdrant clients: {e}")
            
            # Initialize collections in both databases
            self._initialize_collections()
            
        except Exception as e:
            print(f"⚠️  Failed to load config: {e}")
    
    def _initialize_collections(self):
        """Initialize collections for different knowledge types in both databases"""
        if not self.qdrant_cellforge or not self.qdrant_tmp:
            return
        
        collections = [
            'task_analysis_papers',
            'task_analysis_experimental_designs', 
            'task_analysis_implementation_guides',
            'task_analysis_evaluation_frameworks',
            'task_analysis_decision_support'
        ]
        
        # Initialize in CellForge (main database)
        for collection in collections:
            try:
                self.qdrant_cellforge.get_collection(collection)
                print(f"✅ Collection {collection} exists in CellForge")
            except Exception:
                try:
                    self.qdrant_cellforge.create_collection(
                        collection_name=collection,
                        vectors_config=models.VectorParams(
                            size=384,  # all-MiniLM-L6-v2 dimension
                            distance=models.Distance.COSINE
                        )
                    )
                    print(f"✅ Created collection {collection} in CellForge")
                except Exception as e:
                    print(f"⚠️  Failed to create collection {collection} in CellForge: {e}")
        
        # Initialize in cellforge_tmp (temporary database)
        for collection in collections:
            try:
                self.qdrant_tmp.get_collection(collection)
                print(f"✅ Collection {collection} exists in cellforge_tmp")
            except Exception:
                try:
                    self.qdrant_tmp.create_collection(
                        collection_name=collection,
                        vectors_config=models.VectorParams(
                            size=384,  # all-MiniLM-L6-v2 dimension
                            distance=models.Distance.COSINE
                        )
                    )
                    print(f"✅ Created collection {collection} in cellforge_tmp")
                except Exception as e:
                    print(f"⚠️  Failed to create collection {collection} in cellforge_tmp: {e}")
    
    def store_knowledge(self, knowledge_type: str, content: Dict[str, Any], 
                       source: str = "unknown", relevance_score: float = 1.0,
                       metadata: Optional[Dict[str, Any]] = None, 
                       use_main_db: bool = True) -> bool:
        """
        Store knowledge in the knowledge base
        
        Args:
            knowledge_type: Type of knowledge ('papers', 'experimental_designs', etc.)
            content: Knowledge content
            source: Source of the knowledge
            relevance_score: Relevance score (0.0 to 1.0)
            metadata: Additional metadata
            use_main_db: Whether to use CellForge (True) or cellforge_tmp (False)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create knowledge item
            item = KnowledgeItem(
                content=content,
                source=source,
                relevance_score=relevance_score,
                timestamp=datetime.now(),
                metadata=metadata or {},
                knowledge_type=knowledge_type,
                database="CellForge" if use_main_db else "cellforge_tmp"
            )
            
            # Store in memory
            if knowledge_type not in self.knowledge_store:
                self.knowledge_store[knowledge_type] = []
            self.knowledge_store[knowledge_type].append(item)
            
            # Store in Qdrant if available
            if self.encoder:
                if use_main_db and self.qdrant_cellforge:
                    self._store_in_qdrant(item, self.qdrant_cellforge, "CellForge")
                elif not use_main_db and self.qdrant_tmp:
                    self._store_in_qdrant(item, self.qdrant_tmp, "cellforge_tmp")
            
            print(f"✅ Stored {knowledge_type} knowledge from {source} in {'CellForge' if use_main_db else 'cellforge_tmp'}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to store knowledge: {e}")
            return False
    
    def _store_in_qdrant(self, item: KnowledgeItem, qdrant_client: QdrantClient, db_name: str):
        """Store knowledge item in specified Qdrant database"""
        try:
            collection_name = f"task_analysis_{item.knowledge_type}"
            
            # Encode content
            content_str = json.dumps(item.content, ensure_ascii=False)
            embedding = self.encoder.encode(content_str)
            
            # Create point
            point = models.PointStruct(
                id=hash(content_str),
                vector=embedding.tolist(),
                payload={
                    "content": item.content,
                    "source": item.source,
                    "relevance_score": item.relevance_score,
                    "timestamp": item.timestamp.isoformat(),
                    "metadata": item.metadata,
                    "knowledge_type": item.knowledge_type,
                    "database": db_name
                }
            )
            
            # Store to database
            qdrant_client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            
        except Exception as e:
            print(f"⚠️  Failed to store in {db_name}: {e}")
    
    def retrieve_knowledge(self, knowledge_type: str, query: str = "", 
                          limit: int = 10, min_relevance: float = 0.5,
                          use_main_db: bool = True) -> List[KnowledgeItem]:
        """
        Retrieve knowledge from the knowledge base
        
        Args:
            knowledge_type: Type of knowledge to retrieve
            query: Search query (optional, for semantic search)
            limit: Maximum number of results
            min_relevance: Minimum relevance score
            use_main_db: Whether to search in CellForge (True) or cellforge_tmp (False)
            
        Returns:
            List of knowledge items
        """
        try:
            # Get from memory first
            items = self.knowledge_store.get(knowledge_type, [])
            
            # Filter by relevance and database
            items = [item for item in items if item.relevance_score >= min_relevance and 
                    item.database == ("CellForge" if use_main_db else "cellforge_tmp")]
            
            # If query provided and Qdrant available, do semantic search
            if query and self.encoder:
                semantic_items = self._semantic_search(knowledge_type, query, limit, use_main_db)
                if semantic_items:
                    items = semantic_items
            
            # Sort by relevance and limit
            items.sort(key=lambda x: x.relevance_score, reverse=True)
            return items[:limit]
            
        except Exception as e:
            print(f"❌ Failed to retrieve knowledge: {e}")
            return []
    
    def _semantic_search(self, knowledge_type: str, query: str, limit: int, use_main_db: bool) -> List[KnowledgeItem]:
        """Perform semantic search in specified Qdrant database"""
        try:
            collection_name = f"task_analysis_{knowledge_type}"
            qdrant_client = self.qdrant_cellforge if use_main_db else self.qdrant_tmp
            db_name = "CellForge" if use_main_db else "cellforge_tmp"
            
            if not qdrant_client:
                return []
            
            # Encode query
            query_embedding = self.encoder.encode(query)
            
            # Search in Qdrant
            results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit
            )
            
            # Convert to KnowledgeItem
            items = []
            for result in results:
                item = KnowledgeItem(
                    content=result.payload["content"],
                    source=result.payload["source"],
                    relevance_score=result.score,
                    timestamp=datetime.fromisoformat(result.payload["timestamp"]),
                    metadata=result.payload["metadata"],
                    knowledge_type=result.payload["knowledge_type"],
                    database=db_name
                )
                items.append(item)
            
            return items
            
        except Exception as e:
            print(f"⚠️  Semantic search failed in {db_name}: {e}")
            return []
    
    def search_both_databases(self, knowledge_type: str, query: str = "", 
                             limit: int = 10, min_relevance: float = 0.5) -> List[KnowledgeItem]:
        """
        Search in both databases and combine results
        
        Args:
            knowledge_type: Type of knowledge to retrieve
            query: Search query
            limit: Maximum number of results per database
            min_relevance: Minimum relevance score
            
        Returns:
            Combined list of knowledge items from both databases
        """
        try:
            # Search in CellForge
            cellforge_items = self.retrieve_knowledge(
                knowledge_type, query, limit, min_relevance, use_main_db=True
            )
            
            # Search in cellforge_tmp
            tmp_items = self.retrieve_knowledge(
                knowledge_type, query, limit, min_relevance, use_main_db=False
            )
            
            # Combine and sort by relevance
            all_items = cellforge_items + tmp_items
            all_items.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return all_items[:limit * 2]  # Return up to 2x limit items
            
        except Exception as e:
            print(f"❌ Failed to search both databases: {e}")
            return []
    
    def get_knowledge_summary(self, knowledge_type: str = None) -> Dict[str, Any]:
        """
        Get summary of stored knowledge
        
        Args:
            knowledge_type: Specific knowledge type (optional)
            
        Returns:
            Summary dictionary
        """
        try:
            if knowledge_type:
                items = self.knowledge_store.get(knowledge_type, [])
                return {
                    "knowledge_type": knowledge_type,
                    "count": len(items),
                    "sources": list(set(item.source for item in items)),
                    "databases": list(set(item.database for item in items)),
                    "avg_relevance": sum(item.relevance_score for item in items) / len(items) if items else 0.0,
                    "latest_timestamp": max(item.timestamp for item in items).isoformat() if items else None
                }
            else:
                summary = {}
                for kt, items in self.knowledge_store.items():
                    summary[kt] = {
                        "count": len(items),
                        "sources": list(set(item.source for item in items)),
                        "databases": list(set(item.database for item in items)),
                        "avg_relevance": sum(item.relevance_score for item in items) / len(items) if items else 0.0
                    }
                return summary
                
        except Exception as e:
            print(f"❌ Failed to get knowledge summary: {e}")
            return {}
    
    def export_for_mcp(self, knowledge_type: str = None) -> str:
        """
        Export knowledge in MCP-compatible format
        
        Args:
            knowledge_type: Specific knowledge type (optional)
            
        Returns:
            MCP-compatible JSON string
        """
        try:
            if knowledge_type:
                items = self.knowledge_store.get(knowledge_type, [])
            else:
                items = []
                for kt_items in self.knowledge_store.values():
                    items.extend(kt_items)
            
            # Convert to MCP format
            mcp_data = {
                "knowledge_base": {
                    "timestamp": datetime.now().isoformat(),
                    "total_items": len(items),
                    "knowledge_types": list(self.knowledge_store.keys()),
                    "databases": ["CellForge", "cellforge_tmp"],
                    "items": [
                        {
                            "content": item.content,
                            "source": item.source,
                            "relevance_score": item.relevance_score,
                            "knowledge_type": item.knowledge_type,
                            "database": item.database,
                            "metadata": item.metadata
                        }
                        for item in items
                    ]
                }
            }
            
            return json.dumps(mcp_data, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"❌ Failed to export for MCP: {e}")
            return "{}"
    
    def clear_knowledge(self, knowledge_type: str = None, database: str = None):
        """
        Clear knowledge from the knowledge base
        
        Args:
            knowledge_type: Specific knowledge type to clear (optional, clears all if None)
            database: Specific database to clear (optional, clears all if None)
        """
        try:
            if knowledge_type:
                if knowledge_type in self.knowledge_store:
                    if database:
                        self.knowledge_store[knowledge_type] = [
                            item for item in self.knowledge_store[knowledge_type] 
                            if item.database != database
                        ]
                    else:
                        del self.knowledge_store[knowledge_type]
                    print(f"✅ Cleared {knowledge_type} knowledge" + (f" from {database}" if database else ""))
            else:
                if database:
                    for kt in self.knowledge_store:
                        self.knowledge_store[kt] = [
                            item for item in self.knowledge_store[kt] 
                            if item.database != database
                        ]
                    print(f"✅ Cleared all knowledge from {database}")
                else:
                    self.knowledge_store.clear()
                    print("✅ Cleared all knowledge")
                
        except Exception as e:
            print(f"❌ Failed to clear knowledge: {e}")

# Global knowledge base instance
knowledge_base = MCPKnowledgeBase() 