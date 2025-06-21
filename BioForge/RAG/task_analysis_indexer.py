from typing import List, Dict, Any, Optional
import os
import json
import re
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from .utils import TextProcessor

class TaskAnalysisIndexer:
    """
    Vector indexer for storing Task Analysis intermediate results and decisions
    """
    def __init__(self, 
                 qdrant_url: str = "localhost",
                 qdrant_port: int = 6333,
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Task Analysis vector indexer
        
        Args:
            qdrant_url: Qdrant server URL
            qdrant_port: Qdrant server port
            model_name: Name of the sentence transformer model to use
        """
        self.encoder = SentenceTransformer(model_name)
        self.qdrant_client = QdrantClient(qdrant_url, port=qdrant_port)
        self.text_processor = TextProcessor()
        
        # Initialize vector database collections for Task Analysis
        self._initialize_collections()
        
    def _initialize_collections(self) -> None:
        """Initialize Qdrant collections for Task Analysis"""
        collections = [
            'task_analysis_dataparser',
            'task_analysis_dataset_analyst', 
            'task_analysis_problem_investigator',
            'task_analysis_baseline_assessor',
            'task_analysis_refinement',
            'task_analysis_decisions'
        ]
        
        for collection in collections:
            try:
                self.qdrant_client.recreate_collection(
                    collection_name=collection,
                    vectors_config=models.VectorParams(
                        size=384,  # sentence-transformer embedding dimension
                        distance=models.Distance.COSINE
                    )
                )
                print(f"Initialized collection: {collection}")
            except Exception as e:
                print(f"Error initializing collection {collection}: {str(e)}")
    
    def index_dataparser_result(self, dataset_name: str, dataparser_result: Dict[str, Any]) -> None:
        """
        Index DataParser results
        
        Args:
            dataset_name: Name of the dataset
            dataparser_result: DataParser analysis result
        """
        # Extract key information from dataparser result
        content = self._extract_dataparser_content(dataparser_result)
        
        # Encode content
        text_embedding = self.encoder.encode(content)
        
        # Create point
        point = models.PointStruct(
            id=hash(f"dataparser_{dataset_name}_{datetime.now().isoformat()}"),
            vector=text_embedding.tolist(),
            payload={
                "dataset_name": dataset_name,
                "analysis_type": "dataparser",
                "content": content,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "dataset_info": dataparser_result.get("dataset_info", {}),
                    "quality_metrics": dataparser_result.get("quality_metrics", {}),
                    "modality": dataparser_result.get("modality", ""),
                    "perturbation_type": dataparser_result.get("perturbation_type", "")
                },
                "decision_support": {
                    "data_quality_score": dataparser_result.get("quality_metrics", {}).get("overall_score", 0.0),
                    "suitability_for_task": dataparser_result.get("suitability_assessment", ""),
                    "preprocessing_recommendations": dataparser_result.get("preprocessing_recommendations", [])
                }
            }
        )
        
        # Upload to Qdrant
        self.qdrant_client.upsert(
            collection_name="task_analysis_dataparser",
            points=[point]
        )
        
        print(f"Indexed DataParser result for dataset: {dataset_name}")
    
    def index_dataset_analyst_result(self, dataset_name: str, analysis_result: Dict[str, Any]) -> None:
        """
        Index DatasetAnalyst results
        
        Args:
            dataset_name: Name of the dataset
            analysis_result: DatasetAnalyst analysis result
        """
        # Extract key information from dataset analysis
        content = self._extract_dataset_analysis_content(analysis_result)
        
        # Encode content
        text_embedding = self.encoder.encode(content)
        
        # Create point
        point = models.PointStruct(
            id=hash(f"dataset_analyst_{dataset_name}_{datetime.now().isoformat()}"),
            vector=text_embedding.tolist(),
            payload={
                "dataset_name": dataset_name,
                "analysis_type": "dataset_analyst",
                "content": content,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "confidence_score": analysis_result.get("confidence_score", 0.0),
                    "experimental_design": analysis_result.get("content", {}).get("experimental_design", {}),
                    "data_characteristics": analysis_result.get("content", {}).get("data_characteristics", {}),
                    "quality_assessment": analysis_result.get("content", {}).get("quality_assessment", {})
                },
                "decision_support": {
                    "data_suitability": analysis_result.get("content", {}).get("quality_assessment", {}).get("data_suitability", []),
                    "expected_challenges": analysis_result.get("content", {}).get("quality_assessment", {}).get("expected_challenges", []),
                    "preprocessing_considerations": analysis_result.get("content", {}).get("preprocessing", {})
                }
            }
        )
        
        # Upload to Qdrant
        self.qdrant_client.upsert(
            collection_name="task_analysis_dataset_analyst",
            points=[point]
        )
        
        print(f"Indexed DatasetAnalyst result for dataset: {dataset_name}")
    
    def index_problem_investigator_result(self, dataset_name: str, investigation_result: Dict[str, Any]) -> None:
        """
        Index ProblemInvestigator results
        
        Args:
            dataset_name: Name of the dataset
            investigation_result: ProblemInvestigator analysis result
        """
        # Extract key information from problem investigation
        content = self._extract_problem_investigation_content(investigation_result)
        
        # Encode content
        text_embedding = self.encoder.encode(content)
        
        # Create point
        point = models.PointStruct(
            id=hash(f"problem_investigator_{dataset_name}_{datetime.now().isoformat()}"),
            vector=text_embedding.tolist(),
            payload={
                "dataset_name": dataset_name,
                "analysis_type": "problem_investigator",
                "content": content,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "confidence_score": investigation_result.get("confidence_score", 0.0),
                    "formal_definition": investigation_result.get("content", {}).get("formal_definition", {}),
                    "key_challenges": investigation_result.get("content", {}).get("key_challenges", {}),
                    "research_questions": investigation_result.get("content", {}).get("research_questions", {})
                },
                "decision_support": {
                    "task_type": investigation_result.get("content", {}).get("formal_definition", {}).get("task_type", ""),
                    "biological_question": investigation_result.get("content", {}).get("formal_definition", {}).get("biological_question", ""),
                    "evaluation_scenarios": investigation_result.get("content", {}).get("research_questions", {}).get("evaluation_scenarios", []),
                    "evaluation_metrics": investigation_result.get("content", {}).get("analysis_methods", {}).get("evaluation_metrics", [])
                }
            }
        )
        
        # Upload to Qdrant
        self.qdrant_client.upsert(
            collection_name="task_analysis_problem_investigator",
            points=[point]
        )
        
        print(f"Indexed ProblemInvestigator result for dataset: {dataset_name}")
    
    def index_baseline_assessor_result(self, dataset_name: str, assessment_result: Dict[str, Any]) -> None:
        """
        Index BaselineAssessor results
        
        Args:
            dataset_name: Name of the dataset
            assessment_result: BaselineAssessor analysis result
        """
        # Extract key information from baseline assessment
        content = self._extract_baseline_assessment_content(assessment_result)
        
        # Encode content
        text_embedding = self.encoder.encode(content)
        
        # Create point
        point = models.PointStruct(
            id=hash(f"baseline_assessor_{dataset_name}_{datetime.now().isoformat()}"),
            vector=text_embedding.tolist(),
            payload={
                "dataset_name": dataset_name,
                "analysis_type": "baseline_assessor",
                "content": content,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "confidence_score": assessment_result.get("confidence_score", 0.0),
                    "baseline_models": assessment_result.get("content", {}).get("baseline_models", {}),
                    "evaluation_framework": assessment_result.get("content", {}).get("evaluation_framework", {}),
                    "performance_analysis": assessment_result.get("content", {}).get("performance_analysis", {})
                },
                "decision_support": {
                    "sota_model": assessment_result.get("content", {}).get("baseline_models", {}).get("sota", ""),
                    "model_comparison": assessment_result.get("content", {}).get("evaluation_framework", {}).get("model_performance_comparison", []),
                    "current_limitations": assessment_result.get("content", {}).get("performance_analysis", {}).get("current_limitations", []),
                    "improvement_suggestions": assessment_result.get("content", {}).get("improvement_suggestions", {}).get("recommendations", [])
                }
            }
        )
        
        # Upload to Qdrant
        self.qdrant_client.upsert(
            collection_name="task_analysis_baseline_assessor",
            points=[point]
        )
        
        print(f"Indexed BaselineAssessor result for dataset: {dataset_name}")
    
    def index_refinement_result(self, dataset_name: str, refinement_result: Dict[str, Any]) -> None:
        """
        Index RefinementAgent results
        
        Args:
            dataset_name: Name of the dataset
            refinement_result: RefinementAgent analysis result
        """
        # Extract key information from refinement
        content = self._extract_refinement_content(refinement_result)
        
        # Encode content
        text_embedding = self.encoder.encode(content)
        
        # Create point
        point = models.PointStruct(
            id=hash(f"refinement_{dataset_name}_{datetime.now().isoformat()}"),
            vector=text_embedding.tolist(),
            payload={
                "dataset_name": dataset_name,
                "analysis_type": "refinement",
                "content": content,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "refinement_comments": refinement_result.get("refinement_comments", []),
                    "final_recommendations": refinement_result.get("final_recommendations", {})
                },
                "decision_support": {
                    "data_processing": refinement_result.get("final_recommendations", {}).get("data_processing", {}),
                    "model_architecture": refinement_result.get("final_recommendations", {}).get("model_architecture", {}),
                    "training_strategy": refinement_result.get("final_recommendations", {}).get("training_strategy", {}),
                    "evaluation_protocol": refinement_result.get("final_recommendations", {}).get("evaluation_protocol", {}),
                    "implementation_roadmap": refinement_result.get("final_recommendations", {}).get("implementation_roadmap", {})
                }
            }
        )
        
        # Upload to Qdrant
        self.qdrant_client.upsert(
            collection_name="task_analysis_refinement",
            points=[point]
        )
        
        print(f"Indexed RefinementAgent result for dataset: {dataset_name}")
    
    def index_decision(self, dataset_name: str, decision_type: str, decision_content: Dict[str, Any]) -> None:
        """
        Index decision points and their rationale
        
        Args:
            dataset_name: Name of the dataset
            decision_type: Type of decision (e.g., "model_selection", "architecture_choice")
            decision_content: Decision content and rationale
        """
        # Extract decision information
        content = self._extract_decision_content(decision_content)
        
        # Encode content
        text_embedding = self.encoder.encode(content)
        
        # Create point
        point = models.PointStruct(
            id=hash(f"decision_{dataset_name}_{decision_type}_{datetime.now().isoformat()}"),
            vector=text_embedding.tolist(),
            payload={
                "dataset_name": dataset_name,
                "analysis_type": "decision",
                "decision_type": decision_type,
                "content": content,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "decision_rationale": decision_content.get("rationale", ""),
                    "alternatives_considered": decision_content.get("alternatives", []),
                    "confidence_level": decision_content.get("confidence", 0.0)
                },
                "decision_support": {
                    "recommended_action": decision_content.get("recommended_action", ""),
                    "expected_outcome": decision_content.get("expected_outcome", ""),
                    "risk_assessment": decision_content.get("risk_assessment", ""),
                    "implementation_notes": decision_content.get("implementation_notes", [])
                }
            }
        )
        
        # Upload to Qdrant
        self.qdrant_client.upsert(
            collection_name="task_analysis_decisions",
            points=[point]
        )
        
        print(f"Indexed decision for dataset: {dataset_name}, type: {decision_type}")
    
    def search_similar_analyses(self, query: str, analysis_type: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar analyses in the vector database
        
        Args:
            query: Search query
            analysis_type: Type of analysis to search (optional)
            top_k: Number of results to return
            
        Returns:
            List of similar analyses
        """
        # Encode query
        query_embedding = self.encoder.encode(query)
        
        # Determine collection to search
        if analysis_type:
            collection_name = f"task_analysis_{analysis_type}"
        else:
            # Search across all collections
            collections = [
                'task_analysis_dataparser',
                'task_analysis_dataset_analyst', 
                'task_analysis_problem_investigator',
                'task_analysis_baseline_assessor',
                'task_analysis_refinement',
                'task_analysis_decisions'
            ]
            results = []
            for collection in collections:
                try:
                    search_results = self.qdrant_client.search(
                        collection_name=collection,
                        query_vector=query_embedding.tolist(),
                        limit=top_k
                    )
                    for result in search_results:
                        results.append({
                            "collection": collection,
                            "score": result.score,
                            "payload": result.payload
                        })
                except Exception as e:
                    print(f"Error searching collection {collection}: {str(e)}")
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        
        # Search specific collection
        try:
            search_results = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k
            )
            
            return [
                {
                    "collection": collection_name,
                    "score": result.score,
                    "payload": result.payload
                }
                for result in search_results
            ]
        except Exception as e:
            print(f"Error searching collection {collection_name}: {str(e)}")
            return []
    
    def _extract_dataparser_content(self, dataparser_result: Dict[str, Any]) -> str:
        """Extract text content from DataParser result"""
        content_parts = []
        
        if "dataset_info" in dataparser_result:
            dataset_info = dataparser_result["dataset_info"]
            content_parts.append(f"Dataset: {dataset_info.get('dataset_name', '')}")
            content_parts.append(f"Modality: {dataset_info.get('modality', '')}")
            content_parts.append(f"Perturbation Type: {dataset_info.get('perturbation_type', '')}")
        
        if "metadata" in dataparser_result:
            metadata = dataparser_result["metadata"]
            if "basic_info" in metadata:
                basic_info = metadata["basic_info"]
                content_parts.append(f"Title: {basic_info.get('title', '')}")
                content_parts.append(f"Abstract: {basic_info.get('abstract', '')}")
                content_parts.append(f"Organism: {basic_info.get('organism', '')}")
                content_parts.append(f"Tissue: {basic_info.get('tissue', '')}")
                content_parts.append(f"Cell Type: {basic_info.get('cell_type', '')}")
                content_parts.append(f"Method: {basic_info.get('method', '')}")
        
        if "quality_metrics" in dataparser_result:
            quality_metrics = dataparser_result["quality_metrics"]
            content_parts.append(f"Quality Metrics: {json.dumps(quality_metrics)}")
        
        return " ".join(content_parts)
    
    def _extract_dataset_analysis_content(self, analysis_result: Dict[str, Any]) -> str:
        """Extract text content from DatasetAnalyst result"""
        content_parts = []
        
        if "content" in analysis_result:
            content = analysis_result["content"]
            
            if "experimental_design" in content:
                exp_design = content["experimental_design"]
                content_parts.append(f"Biological Objective: {exp_design.get('biological_objective', '')}")
                content_parts.append(f"Technical Approach: {exp_design.get('technical_approach', '')}")
            
            if "data_characteristics" in content:
                data_char = content["data_characteristics"]
                content_parts.append(f"Origin: {data_char.get('origin', '')}")
                content_parts.append(f"Key Features: {json.dumps(data_char.get('key_features', []))}")
                content_parts.append(f"Challenges: {json.dumps(data_char.get('challenges', []))}")
            
            if "quality_assessment" in content:
                quality = content["quality_assessment"]
                content_parts.append(f"Data Suitability: {json.dumps(quality.get('data_suitability', []))}")
                content_parts.append(f"Expected Challenges: {json.dumps(quality.get('expected_challenges', []))}")
        
        return " ".join(content_parts)
    
    def _extract_problem_investigation_content(self, investigation_result: Dict[str, Any]) -> str:
        """Extract text content from ProblemInvestigator result"""
        content_parts = []
        
        if "content" in investigation_result:
            content = investigation_result["content"]
            
            if "formal_definition" in content:
                formal_def = content["formal_definition"]
                content_parts.append(f"Biological Question: {formal_def.get('biological_question', '')}")
                content_parts.append(f"Hypothesis: {formal_def.get('hypothesis_statement', '')}")
                content_parts.append(f"Task Type: {formal_def.get('task_type', '')}")
            
            if "key_challenges" in content:
                challenges = content["key_challenges"]
                content_parts.append(f"Biological Relevance: {json.dumps(challenges.get('biological_relevance', []))}")
                content_parts.append(f"Technical Challenges: {json.dumps(challenges.get('technical_challenges', []))}")
            
            if "research_questions" in content:
                research = content["research_questions"]
                content_parts.append(f"Evaluation Scenarios: {json.dumps(research.get('evaluation_scenarios', []))}")
            
            if "analysis_methods" in content:
                methods = content["analysis_methods"]
                content_parts.append(f"Evaluation Metrics: {json.dumps(methods.get('evaluation_metrics', []))}")
        
        return " ".join(content_parts)
    
    def _extract_baseline_assessment_content(self, assessment_result: Dict[str, Any]) -> str:
        """Extract text content from BaselineAssessor result"""
        content_parts = []
        
        if "content" in assessment_result:
            content = assessment_result["content"]
            
            if "baseline_models" in content:
                models = content["baseline_models"]
                content_parts.append(f"SOTA: {models.get('sota', '')}")
                
                if "detailed_analysis" in models:
                    detailed = models["detailed_analysis"]
                    for model_name, analysis in detailed.items():
                        content_parts.append(f"{model_name}: {json.dumps(analysis.get('shortcomings', []))}")
            
            if "evaluation_framework" in content:
                framework = content["evaluation_framework"]
                content_parts.append(f"Model Comparison: {json.dumps(framework.get('model_performance_comparison', []))}")
            
            if "performance_analysis" in content:
                performance = content["performance_analysis"]
                content_parts.append(f"Current Limitations: {json.dumps(performance.get('current_limitations', []))}")
            
            if "improvement_suggestions" in content:
                improvements = content["improvement_suggestions"]
                content_parts.append(f"Recommendations: {json.dumps(improvements.get('recommendations', []))}")
        
        return " ".join(content_parts)
    
    def _extract_refinement_content(self, refinement_result: Dict[str, Any]) -> str:
        """Extract text content from RefinementAgent result"""
        content_parts = []
        
        if "refinement_comments" in refinement_result:
            comments = refinement_result["refinement_comments"]
            content_parts.append(f"Refinement Comments: {json.dumps(comments)}")
        
        if "final_recommendations" in refinement_result:
            recommendations = refinement_result["final_recommendations"]
            
            if "data_processing" in recommendations:
                data_proc = recommendations["data_processing"]
                content_parts.append(f"Preprocessing Steps: {json.dumps(data_proc.get('preprocessing_steps', []))}")
                content_parts.append(f"Quality Control: {json.dumps(data_proc.get('quality_control', []))}")
            
            if "model_architecture" in recommendations:
                arch = recommendations["model_architecture"]
                content_parts.append(f"Core Components: {json.dumps(arch.get('core_components', []))}")
                content_parts.append(f"Architecture Design: {json.dumps(arch.get('architecture_design', []))}")
            
            if "training_strategy" in recommendations:
                training = recommendations["training_strategy"]
                content_parts.append(f"Training Phases: {json.dumps(training.get('training_phases', []))}")
                content_parts.append(f"Optimization: {json.dumps(training.get('optimization', []))}")
            
            if "evaluation_protocol" in recommendations:
                eval_protocol = recommendations["evaluation_protocol"]
                content_parts.append(f"Evaluation Metrics: {json.dumps(eval_protocol.get('evaluation_metrics', []))}")
                content_parts.append(f"Validation Strategy: {json.dumps(eval_protocol.get('validation_strategy', []))}")
            
            if "implementation_roadmap" in recommendations:
                roadmap = recommendations["implementation_roadmap"]
                content_parts.append(f"Implementation Roadmap: {json.dumps(roadmap)}")
        
        return " ".join(content_parts)
    
    def _extract_decision_content(self, decision_content: Dict[str, Any]) -> str:
        """Extract text content from decision"""
        content_parts = []
        
        content_parts.append(f"Decision Type: {decision_content.get('decision_type', '')}")
        content_parts.append(f"Rationale: {decision_content.get('rationale', '')}")
        content_parts.append(f"Recommended Action: {decision_content.get('recommended_action', '')}")
        content_parts.append(f"Expected Outcome: {decision_content.get('expected_outcome', '')}")
        content_parts.append(f"Risk Assessment: {decision_content.get('risk_assessment', '')}")
        content_parts.append(f"Alternatives Considered: {json.dumps(decision_content.get('alternatives', []))}")
        content_parts.append(f"Implementation Notes: {json.dumps(decision_content.get('implementation_notes', []))}")
        
        return " ".join(content_parts) 