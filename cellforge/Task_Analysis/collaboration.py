from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os

try:
    from .data_structures import AnalysisResult, TaskAnalysisReport
    from .dataset_analyst import DatasetAnalyst
    from .problem_investigator import ProblemInvestigator
    from .baseline_assessor import BaselineAssessor
    from .refinement_agent import RefinementAgent
    from .rag import RAGSystem
    from ..llm import LLMInterface
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_structures import AnalysisResult, TaskAnalysisReport
    from dataset_analyst import DatasetAnalyst
    from problem_investigator import ProblemInvestigator
    from baseline_assessor import BaselineAssessor
    from refinement_agent import RefinementAgent
    from rag import RAGSystem
    from cellforge.llm import LLMInterface

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
        
        rag_results = self.rag_system.search(task_description)  
        
        
        dataset_analysis = self.dataset_analyst.analyze_dataset(
            task_description,
            dataset_info,
            rag_results  
        )
        
        problem_investigation = self.problem_investigator.investigate_problem(
            task_description,
            dataset_info,
            rag_results  
        )
        
        baseline_assessment = self.baseline_assessor.assess_baselines(
            task_description,
            dataset_info,
            rag_results  
        )
        
        
        
        final_report = self.refinement_agent.refine_analysis(
            dataset_analysis,
            problem_investigation,
            baseline_assessment
        )
        
        
        self._save_report(final_report, task_description, dataset_info)
        
        return final_report
    
    def _save_report(self, report: TaskAnalysisReport, task_description: str, dataset_info: Dict[str, Any]):
        """Save the analysis report to JSON file"""
        try:
            from pathlib import Path
            import json
            from datetime import datetime
            
            
            results_dir = Path("cellforge/Task_Analysis/results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            
            filename = f"task_analysis_{timestamp}.json"
            file_path = results_dir / filename
            
            
            report_data = {
                "timestamp": timestamp,
                "task_description": task_description,
                "dataset_info": dataset_info,
                "task_type": "gene_knockout",  
                "analysis_results": {
                    "dataset_analysis": report.dataset_analysis.to_dict() if hasattr(report.dataset_analysis, 'to_dict') else str(report.dataset_analysis),
                    "problem_investigation": report.problem_investigation.to_dict() if hasattr(report.problem_investigation, 'to_dict') else str(report.problem_investigation),
                    "baseline_assessment": report.baseline_assessment.to_dict() if hasattr(report.baseline_assessment, 'to_dict') else str(report.baseline_assessment),
                    "refined_analysis": report.refined_analysis.to_dict() if hasattr(report.refined_analysis, 'to_dict') else str(report.refined_analysis)
                },
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
            
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Task analysis report saved to: {file_path}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save task analysis report: {e}")
            
    
    def _retrieve_relevant_papers(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant papers from vector database with single cell focus
        
        Args:
            query: Search query
            top_k: Number of papers to retrieve
            
        Returns:
            List of relevant papers with metadata
        """
        
        rag_results = self.rag_system.search(query)  
        
        
        papers = []
        for result in rag_results[:top_k]:  
            papers.append({
                "title": result.get("title", ""),
                "abstract": result.get("content", result.get("snippet", "")),  
                "metadata": result.get("metadata", {}),
                "score": result.get("score", 0.0)
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
            
            try:
                return json.loads(response["content"])
            except json.JSONDecodeError:
                
                import re
                content = response["content"]
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                else:
                    
                    return {"content": content, "error": "Failed to parse JSON response"}
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")
