import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
from github import Github
from serpapi import GoogleSearch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from datetime import datetime
import json
from .utils import TextProcessor
from .parser import PaperParser
from .indexer import VectorIndexer
from .search import HybridSearcher, SearchResult

@dataclass
class SearchResult:
    """Structured search result for single cell perturbation analysis"""
    title: str
    content: str
    source: str
    url: str
    relevance_score: float
    publication_date: Optional[datetime]
    citations: Optional[int]
    metadata: Dict[str, Any] = None  # Add metadata field

class RAGSystem:
    """
    Advanced Retrieval-Augmented Generation system for single cell perturbation analysis.
    Implements hybrid BFS-DFS search strategy with multiple data sources.
    """
    def __init__(self, qdrant_url: str = "localhost", qdrant_port: int = 6333):
        
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        self.pubmed_api_key = os.getenv("PUBMED_API_KEY")
        self.pubmed_tool = os.getenv("PUBMED_TOOL", "BioForge")
        self.pubmed_email = os.getenv("PUBMED_EMAIL")
        
        
        self.text_processor = TextProcessor()
        self.paper_parser = PaperParser()
        self.vector_indexer = VectorIndexer(qdrant_url, qdrant_port)
        self.hybrid_searcher = HybridSearcher(qdrant_url, qdrant_port)
        
        
        if self.github_token and self.github_token != "your_github_token_here":
            self.github_client = Github(self.github_token)
        else:
            self.github_client = None
            print("GitHub token not configured, GitHub search will be disabled")
        
        
        self._load_single_cell_terms()
        
    def _load_single_cell_terms(self) -> None:
        """Load specialized terms related to single-cell perturbation analysis"""
        self.single_cell_terms = {
            "perturbation_types": [
                "gene_knockout", "gene_knockdown", "CRISPR", "RNAi",
                "drug_treatment", "small_molecule", "compound",
                "overexpression", "transfection", "transduction"
            ],
            "technologies": [
                "scRNA-seq", "single-cell RNA sequencing", "10x Genomics",
                "Drop-seq", "Smart-seq2", "CEL-seq2", "sci-RNA-seq",
                "Perturb-seq", "CROP-seq", "Mosaic-seq"
            ],
            "analysis_methods": [
                "differential_expression", "trajectory_analysis",
                "pseudotime", "cell_type_annotation", "clustering",
                "dimensionality_reduction", "batch_correction",
                "integration", "regulatory_network"
            ],
            "cell_types": [
                "T_cell", "B_cell", "macrophage", "dendritic_cell",
                "stem_cell", "neuron", "fibroblast", "epithelial_cell"
            ],
            "perturbation_effects": [
                "cell_state_change", "differentiation", "proliferation",
                "apoptosis", "cell_cycle", "signaling_pathway",
                "transcriptional_regulation", "epigenetic_modification"
            ]
        }
        
    def search(self, task_description: str, dataset_info: Dict[str, Any]) -> Dict[str, List[SearchResult]]:
        """
        Execute hybrid search strategy with single cell perturbation focus and decision support
        
        Args:
            task_description: Description of the search task
            dataset_info: Information about the dataset
            
        Returns:
            Dictionary containing search results from different sources with decision support
        """
        
        # Extract keywords and add single-cell perturbation related terms
        keywords = self._extract_keywords(task_description, dataset_info)
        
        # Perform hybrid search
        results = self.hybrid_searcher.search(keywords)
        
        # Add metadata
        for source, search_results in results.items():
            for result in search_results:
                result.metadata = {
                    "perturbation_type": self._identify_perturbation_type(result.content),
                    "technology": self._identify_technology(result.content),
                    "analysis_method": self._identify_analysis_method(result.content),
                    "cell_type": self._identify_cell_type(result.content),
                    "perturbation_effect": self._identify_perturbation_effect(result.content),
                    "decision_support": self._extract_decision_support(result)
                }
        
        # Get decision recommendations
        decision_recommendations = self.hybrid_searcher.get_decision_recommendations(task_description, dataset_info)
        results["decision_recommendations"] = decision_recommendations
        
        return results
    
    def _extract_keywords(self, task_description: str, dataset_info: Dict[str, Any]) -> List[str]:
        """
        Extract relevant keywords with single cell perturbation focus
        
        Args:
            task_description: Description of the search task
            dataset_info: Information about the dataset
            
        Returns:
            List of extracted keywords
        """
        
        # Combine task description and dataset information
        text = f"{task_description} {dataset_info.get('name', '')} {dataset_info.get('modality', '')}"
        
        # Extract technical terms
        technical_terms = self.text_processor.extract_technical_terms(text)
        
        # Extract biological terms
        biological_terms = self.text_processor.extract_biological_terms(text)
        
        # Extract keywords
        keywords = self.text_processor.extract_keywords(text)
        
        
        single_cell_terms = []
        for category, terms in self.single_cell_terms.items():
            single_cell_terms.extend(terms)
        
        
        all_terms = list(set(technical_terms + biological_terms + keywords + single_cell_terms))
        
        return all_terms
    
    def _identify_perturbation_type(self, text: str) -> List[str]:
        """Identify perturbation types in text"""
        return [term for term in self.single_cell_terms["perturbation_types"] 
                if term.lower() in text.lower()]
    
    def _identify_technology(self, text: str) -> List[str]:
        """Identify technology types in text"""
        return [term for term in self.single_cell_terms["technologies"] 
                if term.lower() in text.lower()]
    
    def _identify_analysis_method(self, text: str) -> List[str]:
        """Identify analysis methods in text"""
        return [term for term in self.single_cell_terms["analysis_methods"] 
                if term.lower() in text.lower()]
    
    def _identify_cell_type(self, text: str) -> List[str]:
        """Identify cell types in text"""
        return [term for term in self.single_cell_terms["cell_types"] 
                if term.lower() in text.lower()]
    
    def _identify_perturbation_effect(self, text: str) -> List[str]:
        """Identify perturbation effects in text"""
        return [term for term in self.single_cell_terms["perturbation_effects"] 
                if term.lower() in text.lower()]
    
    def index_papers(self, papers_dir: str) -> None:
        """
        Index papers from a directory with single cell focus
        
        Args:
            papers_dir: Directory containing paper PDFs
        """
        self.vector_indexer.index_papers(papers_dir)
    
    def index_code(self, code_dir: str) -> None:
        """
        Index code from a directory with single cell focus
        
        Args:
            code_dir: Directory containing code files
        """
        self.vector_indexer.index_code(code_dir)
    
    def get_paper_summary(self, paper_path: str) -> Dict[str, Any]:
        """
        Get summary of a paper with single cell focus
        
        Args:
            paper_path: Path to the paper PDF
            
        Returns:
            Dictionary containing paper summary
        """
        summary = self.paper_parser.parse_paper(paper_path)
        
        
        summary["single_cell_metadata"] = {
            "perturbation_type": self._identify_perturbation_type(summary["content"]),
            "technology": self._identify_technology(summary["content"]),
            "analysis_method": self._identify_analysis_method(summary["content"]),
            "cell_type": self._identify_cell_type(summary["content"]),
            "perturbation_effect": self._identify_perturbation_effect(summary["content"])
        }
        
        return summary
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts with single cell focus
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        return self.text_processor.calculate_text_similarity(text1, text2)
    
    def extract_summary(self, text: str) -> str:
        """
        Extract summary from text with single cell focus
        
        Args:
            text: Input text
            
        Returns:
            Extracted summary
        """
        return self.text_processor.extract_summary(text)
    
    def get_related_papers(self, paper_id: str) -> List[SearchResult]:
        """
        Get related papers based on single cell perturbation analysis
        
        Args:
            paper_id: ID of the paper
            
        Returns:
            List of related papers
        """
        return self.hybrid_searcher.get_related_papers(paper_id)
    
    def get_related_code(self, paper_id: str) -> List[SearchResult]:
        """
        Get related code implementations based on single cell perturbation analysis
        
        Args:
            paper_id: ID of the reference paper
            
        Returns:
            List of related code implementations
        """
        return self.hybrid_searcher.get_related_code(paper_id)
    
    def get_decision_support(self, task_description: str, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get specialized decision support information
        
        Args:
            task_description: Task description
            dataset_info: Dataset information
            
        Returns:
            Decision support information
        """
        return self.hybrid_searcher.get_decision_recommendations(task_description, dataset_info)
    
    def search_experimental_designs(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Search for experimental design information
        
        Args:
            task_description: Task description
            
        Returns:
            List of experimental design information
        """
        
        design_query = f"experimental design {task_description}"
        query_vector = self.hybrid_searcher.encoder.encode(design_query)
        
        search_result = self.hybrid_searcher.qdrant_client.search(
            collection_name="papers",
            query_vector=query_vector,
            limit=5,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="type",
                        match=models.MatchValue(value="experimental_design")
                    )
                ]
            )
        )
        
        designs = []
        for hit in search_result:
            designs.append({
                "title": hit.payload.get("title", ""),
                "content": hit.payload.get("content", ""),
                "metadata": hit.payload.get("metadata", {}),
                "score": hit.score
            })
        
        return designs
    
    def search_evaluation_frameworks(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Search for evaluation framework information
        
        Args:
            task_description: Task description
            
        Returns:
            List of evaluation framework information
        """
        
        framework_query = f"evaluation framework {task_description}"
        query_vector = self.hybrid_searcher.encoder.encode(framework_query)
        
        search_result = self.hybrid_searcher.qdrant_client.search(
            collection_name="papers",
            query_vector=query_vector,
            limit=5,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="type",
                        match=models.MatchValue(value="evaluation_framework")
                    )
                ]
            )
        )
        
        frameworks = []
        for hit in search_result:
            frameworks.append({
                "title": hit.payload.get("title", ""),
                "content": hit.payload.get("content", ""),
                "metadata": hit.payload.get("metadata", {}),
                "score": hit.score
            })
        
        return frameworks
    
    def search_implementation_guides(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Search for implementation guide information
        
        Args:
            task_description: Task description
            
        Returns:
            List of implementation guide information
        """
        
        guide_query = f"implementation guide {task_description}"
        query_vector = self.hybrid_searcher.encoder.encode(guide_query)
        
        search_result = self.hybrid_searcher.qdrant_client.search(
            collection_name="papers",
            query_vector=query_vector,
            limit=5,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="type",
                        match=models.MatchValue(value="implementation_guide")
                    )
                ]
            )
        )
        
        guides = []
        for hit in search_result:
            guides.append({
                "title": hit.payload.get("title", ""),
                "content": hit.payload.get("content", ""),
                "metadata": hit.payload.get("metadata", {}),
                "score": hit.score
            })
        
        return guides
