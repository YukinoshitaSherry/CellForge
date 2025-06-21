from typing import List, Dict, Any, Optional
import os
import json
import re
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from .parser import PaperParser
from .utils import TextProcessor

class VectorIndexer:
    """
    Vector indexer for vectorizing and storing papers and code in Qdrant
    """
    def __init__(self, 
                 qdrant_url: str = "localhost",
                 qdrant_port: int = 6333,
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector indexer
        
        Args:
            qdrant_url: Qdrant server URL
            qdrant_port: Qdrant server port
            model_name: Name of the sentence transformer model to use
        """
        self.encoder = SentenceTransformer(model_name)
        self.qdrant_client = QdrantClient(qdrant_url, port=qdrant_port)
        self.paper_parser = PaperParser()
        self.text_processor = TextProcessor()
        
        
        self._initialize_collections()
        
    def _initialize_collections(self) -> None:
        """Initialize Qdrant collections"""
        collections = ['papers', 'code', 'references']
        
        for collection in collections:
            self.qdrant_client.recreate_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=384,  
                    distance=models.Distance.COSINE
                )
            )
    
    def index_papers(self, papers_dir: str) -> None:
        """
        Index papers directory
        
        Args:
            papers_dir: Path to papers directory
        """
        
        for root, _, files in os.walk(papers_dir):
            for file in files:
                if file.endswith('.pdf'):
                    
                    paper_path = os.path.join(root, file)
                    try:
                        paper_content = self.paper_parser.parse_pdf(paper_path)
                        self._index_paper(paper_content)
                    except Exception as e:
                        print(f"Error indexing paper {paper_path}: {str(e)}")
    
    def index_code(self, code_dir: str) -> None:
        """
        Index code directory
        
        Args:
            code_dir: Path to code directory
        """
        
        for root, _, files in os.walk(code_dir):
            for file in files:
                if file.endswith(('.py', '.ipynb')):
                    
                    code_path = os.path.join(root, file)
                    try:
                        with open(code_path, 'r', encoding='utf-8') as f:
                            code_content = f.read()
                        self._index_code(code_path, code_content)
                    except Exception as e:
                        print(f"Error indexing code {code_path}: {str(e)}")
    
    def _index_paper(self, paper_content: Dict[str, Any]) -> None:
        """
        Index single paper with enhanced decision support metadata
        
        Args:
            paper_content: Paper content dictionary
        """
        
        text_content = self._extract_paper_text(paper_content)
        
        
        text_embedding = self.encoder.encode(text_content)
        
        
        decision_metadata = self._extract_decision_metadata(paper_content)
        
        
        point = models.PointStruct(
            id=hash(paper_content.get("title", "")),
            vector=text_embedding.tolist(),
            payload={
                "title": paper_content.get("title", ""),
                "authors": paper_content.get("authors", []),
                "publication_date": paper_content.get("publication_date"),
                "journal": paper_content.get("journal", ""),
                "abstract": paper_content.get("abstract", ""),
                "content": text_content,
                "source": "Local PDF",
                "url": paper_content.get("url", ""),
                "citations": paper_content.get("citations"),
                "metadata": paper_content.get("metadata", {}),
                "decision_support": decision_metadata
            }
        )
        
        
        self.qdrant_client.upsert(
            collection_name="papers",
            points=[point]
        )
        
        
        self._index_references(paper_content.get("references", []))
    
    def _index_code(self, code_path: str, code_content: str) -> None:
        """
        Index code file
        
        Args:
            code_path: Path to code file
            code_content: Code content
        """
        
        processed_code = self._preprocess_code(code_content)
        
        
        code_embedding = self.encoder.encode(processed_code)
        
        
        point = models.PointStruct(
            id=hash(code_path),
            vector=code_embedding.tolist(),
            payload={
                "title": os.path.basename(code_path),
                "content": processed_code,
                "source": "Local Code",
                "url": code_path,
                "metadata": {
                    "file_type": os.path.splitext(code_path)[1],
                    "last_modified": datetime.fromtimestamp(
                        os.path.getmtime(code_path)
                    ).isoformat()
                }
            }
        )
        
        
        self.qdrant_client.upsert(
            collection_name="code",
            points=[point]
        )
    
    def _index_references(self, references: List[Dict[str, Any]]) -> None:
        """
        Index references
        
        Args:
            references: List of references
        """
        for ref in references:
            
            ref_text = ref.get("text", "")
            
            
            ref_embedding = self.encoder.encode(ref_text)
            
            
            point = models.PointStruct(
                id=hash(ref_text),
                vector=ref_embedding.tolist(),
                payload={
                    "text": ref_text,
                    "authors": ref.get("authors", []),
                    "year": ref.get("year"),
                    "title": ref.get("title", ""),
                    "journal": ref.get("journal", ""),
                    "doi": ref.get("doi", ""),
                    "metadata": ref.get("metadata", {})
                }
            )
            
            
            self.qdrant_client.upsert(
                collection_name="references",
                points=[point]
            )
    
    def _extract_paper_text(self, paper_content: Dict[str, Any]) -> str:
        """
        Extract paper text content
        
        Args:
            paper_content: Paper content dictionary
            
        Returns:
            Extracted text content
        """
        
        text_parts = [
            paper_content.get("title", ""),
            paper_content.get("abstract", ""),
            paper_content.get("introduction", ""),
            paper_content.get("methods", ""),
            paper_content.get("results", ""),
            paper_content.get("discussion", ""),
            paper_content.get("conclusion", "")
        ]
        
        return " ".join(filter(None, text_parts))
    
    def _preprocess_code(self, code_content: str) -> str:
        """
        Preprocess code content
        
        Args:
            code_content: Code content
            
        Returns:
            Preprocessed code content
        """
        
        code_content = re.sub(r'#.*', '', code_content)
        code_content = re.sub(r'""".*?"""', '', code_content, flags=re.DOTALL)
        code_content = re.sub(r"'''.*?'''", '', code_content, flags=re.DOTALL)
        
        
        code_content = re.sub(r'\n\s*\n', '\n', code_content)
        
        return code_content.strip()
    
    def _extract_decision_metadata(self, paper_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract decision support metadata
        
        Args:
            paper_content: Paper content dictionary
            
        Returns:
            Decision support metadata
        """
        metadata = paper_content.get("metadata", {})
        decision_support = metadata.get("decision_support", {})
        
        
        if not decision_support:
            text_content = self._extract_paper_text(paper_content)
            decision_support = self._extract_decision_info_from_text(text_content)
        
        return decision_support
    
    def _extract_decision_info_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract decision support information from text
        
        Args:
            text: Text content
            
        Returns:
            Decision support information
        """
        decision_info = {
            "model_recommendations": [],
            "evaluation_metrics": [],
            "data_requirements": [],
            "implementation_complexity": "unknown",
            "biological_interpretability": "unknown"
        }
        
        
        model_keywords = [
            "transformer", "graph neural network", "autoencoder", "variational autoencoder",
            "convolutional neural network", "recurrent neural network", "attention mechanism",
            "foundation model", "pretrained model", "contrastive learning"
        ]
        
        for keyword in model_keywords:
            if keyword.lower() in text.lower():
                decision_info["model_recommendations"].append(keyword)
        
        
        metric_keywords = [
            "pearson correlation", "spearman correlation", "mean squared error", "mse",
            "accuracy", "precision", "recall", "f1 score", "auc", "roc",
            "silhouette score", "adjusted rand index", "normalized mutual information"
        ]
        
        for keyword in metric_keywords:
            if keyword.lower() in text.lower():
                decision_info["evaluation_metrics"].append(keyword)
        
        
        data_keywords = [
            "baseline expression", "perturbation metadata", "gene network",
            "chemical structure", "dose response", "cell type annotation",
            "quality control", "batch correction", "normalization"
        ]
        
        for keyword in data_keywords:
            if keyword.lower() in text.lower():
                decision_info["data_requirements"].append(keyword)
        
        
        complexity_indicators = {
            "very_high": ["foundation model", "large scale", "pretrained"],
            "high": ["transformer", "graph neural network", "attention"],
            "moderate": ["autoencoder", "variational", "representation learning"],
            "low": ["linear", "regression", "classification"]
        }
        
        for complexity, indicators in complexity_indicators.items():
            if any(indicator.lower() in text.lower() for indicator in indicators):
                decision_info["implementation_complexity"] = complexity
                break
        
        
        interpretability_indicators = {
            "very_high": ["interpretable", "explainable", "latent space"],
            "high": ["attention", "feature importance", "pathway"],
            "moderate": ["correlation", "prediction", "model"],
            "low": ["black box", "complex", "opaque"]
        }
        
        for interpretability, indicators in interpretability_indicators.items():
            if any(indicator.lower() in text.lower() for indicator in indicators):
                decision_info["biological_interpretability"] = interpretability
                break
        
        return decision_info 