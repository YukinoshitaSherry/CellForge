from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os

from .data_structures import AnalysisResult, TaskAnalysisReport
from .dataset_analyst import DatasetAnalyst
from .problem_investigator import ProblemInvestigator
from .baseline_assessor import BaselineAssessor
from .refinement_agent import RefinementAgent
from .rag import RAGSystem
from .llm import LLMInterface

@dataclass
class Agent:
    """Structured agent definition for single cell perturbation prediction"""
    name: str
    role: str
    expertise: List[str]
    prompt: str
    confidence_threshold: float = 0.8

@dataclass
class AnalysisResult:
    """Structured analysis result with single cell focus"""
    content: Dict[str, Any]
    confidence_score: float
    timestamp: datetime
    metadata: Dict[str, Any]

class CollaborationSystem:
    """
    Advanced multi-agent collaboration system for single cell perturbation prediction analysis.
    Implements graph-based discussion with confidence scoring and RAG integration.
    """
    def __init__(self, qdrant_url: str = "localhost", qdrant_port: int = 6333):
        
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        
        self.qdrant_client = QdrantClient(url=qdrant_url, port=qdrant_port)
        
        
        self.rag_system = RAGSystem(qdrant_url, qdrant_port)
        
        self.llm = LLMInterface()
        
        self.dataset_analyst = DatasetAnalyst(self.rag_system)
        self.problem_investigator = ProblemInvestigator(self.rag_system)
        self.baseline_assessor = BaselineAssessor(self.rag_system)
        self.refinement_agent = RefinementAgent(self.rag_system)
        
    def run_analysis(self, task_description: str, dataset_info: Dict[str, Any]) -> TaskAnalysisReport:
        """
        Run collaborative analysis with multiple agents for single cell perturbation prediction
        
        Args:
            task_description: Description of the task
            dataset_info: Information about the dataset
            
        Returns:
            Comprehensive analysis report
        """
        
        rag_results = self.rag_system.search(task_description, dataset_info)
        
        
        dataset_analysis = self.dataset_analyst.analyze_dataset(
            task_description,
            dataset_info,
            rag_results.get("papers", [])
        )
        
        problem_investigation = self.problem_investigator.investigate_problem(
            task_description,
            dataset_info,
            rag_results.get("papers", [])
        )
        
        baseline_assessment = self.baseline_assessor.assess_baselines(
            task_description,
            dataset_info,
            rag_results.get("papers", [])
        )
        
        
        final_report = self.refinement_agent.refine_analysis(
            dataset_analysis,
            problem_investigation,
            baseline_assessment,
            rag_results
        )
        
        return final_report
    
    def _retrieve_relevant_papers(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant papers from vector database with single cell focus
        
        Args:
            query: Search query
            top_k: Number of papers to retrieve
            
        Returns:
            List of relevant papers with metadata
        """
        
        rag_results = self.rag_system.search(query, {})
        
        
        papers = []
        for result in rag_results.get("papers", [])[:top_k]:
            papers.append({
                "title": result.title,
                "abstract": result.content,
                "metadata": result.metadata,
                "score": result.relevance_score
            })
        
        return papers
    
    def _run_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Run LLM to generate analysis content with single cell focus
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated content in JSON format
        """
        system_prompt = "You are an expert in single-cell perturbation analysis. Provide your response in valid JSON format."
        
        try:
            response = self.llm.generate(prompt, system_prompt)
            return response
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")
