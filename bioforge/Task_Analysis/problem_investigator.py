from typing import Dict, Any, List
from datetime import datetime
import os
import json

from .data_structures import AnalysisResult
from .rag import RAGSystem
from .llm import LLMInterface

class ProblemInvestigator:
    """Expert agent for investigating research problems and designing solution approaches"""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        
        
        self.llm = LLMInterface()
    
    def _run_llm(self, prompt: str) -> Dict[str, Any]:
        system_prompt = "You are an expert in single-cell perturbation research. Provide your response in valid JSON format."
        
        try:
            
            response = self.llm.generate(prompt, system_prompt)
            return response
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")
    
    def investigate_problem(self, task_description: str, dataset_info: Dict[str, Any],
                          retrieved_papers: List[Dict[str, Any]]) -> AnalysisResult:
        """
        Investigate research problem and design solution approach
        
        Args:
            task_description: Description of the research task
            dataset_info: Dictionary containing dataset metadata
            retrieved_papers: List of relevant papers from vector database
            
        Returns:
            AnalysisResult with problem investigation
        """
        
        rag_results = self.rag_system.search(task_description, dataset_info)
        
        
        decision_support = self.rag_system.get_decision_support(task_description, dataset_info)
        
        
        experimental_designs = self.rag_system.search_experimental_designs(task_description)
        implementation_guides = self.rag_system.search_implementation_guides(task_description)
        
        
        all_papers = retrieved_papers + [
            {
                "title": result.title,
                "abstract": result.content,
                "metadata": result.metadata,
                "decision_support": result.metadata.get("decision_support", {})
            }
            for result in rag_results.get("papers", [])
        ]
        
        
        prompt = self._format_prompt_with_decision_support(
            task_description, dataset_info, all_papers, decision_support, 
            experimental_designs, implementation_guides
        )
        
        
        investigation_content = self._run_llm(prompt)
        
        return AnalysisResult(
            content=investigation_content,
            confidence_score=1.0,  
            timestamp=datetime.now(),
            metadata={
                "agent": "Problem Investigator",
                "rag_results": {
                    "papers_count": len(rag_results.get("papers", [])),
                    "decision_support_available": bool(decision_support),
                    "experimental_designs_count": len(experimental_designs),
                    "implementation_guides_count": len(implementation_guides)
                },
                "decision_support": decision_support
            }
        )
    
    def _format_prompt(self, task_description: str, dataset_info: Dict[str, Any],
                      retrieved_papers: List[Dict[str, Any]]) -> str:
        """Format prompt for problem investigation"""
        papers_context = "\n".join([
            f"- {paper['title']}: {paper['abstract'][:200]}..."
            for paper in retrieved_papers[:5]
        ])
        
        return f"""You are an expert in biological research problem analysis with extensive experience in designing computational solutions for biological applications. Your task is to provide a comprehensive investigation of the research problem and design a solution approach, focusing on problem definition, key challenges, research questions, and analysis methods.

1. Define Research Problem:
   - Formally define the problem in biological context
   - Identify key biological variables and their relationships
   - Specify input-output mappings and biological constraints
   - Define evaluation criteria with biological significance
   - Establish success metrics with biological validation

2. Analyze Key Challenges:
   - Identify biological and technical challenges
   - Assess data quality and biological variability
   - Evaluate computational complexity and scalability
   - Consider biological interpretability requirements
   - Address validation and reproducibility concerns

3. Formulate Research Questions:
   - Define primary research questions with biological focus
   - Identify key hypotheses to be tested
   - Specify biological mechanisms to be investigated
   - Outline experimental validation requirements
   - Establish biological significance criteria

4. Design Analysis Methods:
   - Propose computational approaches with biological relevance
   - Design validation strategies with biological context
   - Specify analysis pipelines with biological interpretation
   - Plan experimental validation with biological controls
   - Establish reproducibility standards with biological validation

Task Description:
{task_description}

Dataset Information:
{json.dumps(dataset_info, indent=2)}

Relevant Literature:
{papers_context}

Please provide a comprehensive investigation in the following JSON format:
{{
    "problem_definition": {{
        "formal_definition": {{
            "biological_context": "string",
            "input_output_mapping": "string",
            "biological_constraints": ["string"],
            "evaluation_criteria": ["string"],
            "success_metrics": ["string"]
        }},
        "key_variables": {{
            "biological": ["string"],
            "technical": ["string"],
            "relationships": ["string"],
            "constraints": ["string"],
            "validation_requirements": ["string"]
        }},
        "scope": {{
            "biological_scope": "string",
            "technical_scope": "string",
            "limitations": ["string"],
            "assumptions": ["string"],
            "biological_validation": "string"
        }}
    }},
    "key_challenges": {{
        "biological": {{
            "challenges": ["string"],
            "impact": "string",
            "mitigation": ["string"],
            "validation": "string",
            "biological_considerations": "string"
        }},
        "technical": {{
            "challenges": ["string"],
            "impact": "string",
            "mitigation": ["string"],
            "validation": "string",
            "implementation_requirements": "string"
        }},
        "data_quality": {{
            "issues": ["string"],
            "impact": "string",
            "mitigation": ["string"],
            "validation": "string",
            "biological_validation": "string"
        }},
        "computational": {{
            "challenges": ["string"],
            "impact": "string",
            "mitigation": ["string"],
            "validation": "string",
            "resource_requirements": "string"
        }},
        "interpretability": {{
            "requirements": ["string"],
            "challenges": ["string"],
            "solutions": ["string"],
            "validation": "string",
            "biological_validation": "string"
        }}
    }},
    "research_questions": {{
        "primary": {{
            "questions": ["string"],
            "hypotheses": ["string"],
            "biological_significance": "string",
            "validation_approach": "string",
            "expected_outcomes": ["string"]
        }},
        "secondary": {{
            "questions": ["string"],
            "hypotheses": ["string"],
            "biological_significance": "string",
            "validation_approach": "string",
            "expected_outcomes": ["string"]
        }},
        "biological_mechanisms": {{
            "mechanisms": ["string"],
            "investigation_approach": "string",
            "validation_methods": ["string"],
            "expected_insights": ["string"],
            "biological_validation": "string"
        }},
        "experimental_validation": {{
            "requirements": ["string"],
            "methods": ["string"],
            "controls": ["string"],
            "metrics": ["string"],
            "biological_validation": "string"
        }}
    }},
    "analysis_methods": {{
        "computational_approaches": {{
            "methods": ["string"],
            "rationale": "string",
            "implementation": "string",
            "validation": "string",
            "biological_validation": "string"
        }},
        "validation_strategies": {{
            "strategies": ["string"],
            "rationale": "string",
            "implementation": "string",
            "metrics": ["string"],
            "biological_validation": "string"
        }},
        "analysis_pipelines": {{
            "pipelines": ["string"],
            "components": ["string"],
            "workflow": "string",
            "validation": "string",
            "biological_interpretation": "string"
        }},
        "experimental_validation": {{
            "methods": ["string"],
            "controls": ["string"],
            "metrics": ["string"],
            "analysis": "string",
            "biological_validation": "string"
        }},
        "reproducibility": {{
            "standards": ["string"],
            "requirements": ["string"],
            "validation": "string",
            "documentation": "string",
            "biological_validation": "string"
        }}
    }}
}}"""

    def _format_prompt_with_decision_support(self, task_description: str, dataset_info: Dict[str, Any],
                                            papers: List[Dict[str, Any]], decision_support: Dict[str, Any],
                                            experimental_designs: List[Dict[str, Any]], implementation_guides: List[Dict[str, Any]]) -> str:
        """Format prompt for problem investigation with decision support"""
        papers_context = "\n".join([
            f"- {paper['title']}: {paper['abstract'][:200]}..."
            for paper in papers[:5]
        ])
        
        return f"""You are an expert in biological research problem analysis with extensive experience in designing computational solutions for biological applications. Your task is to provide a comprehensive investigation of the research problem and design a solution approach, focusing on problem definition, key challenges, research questions, and analysis methods.

1. Define Research Problem:
   - Formally define the problem in biological context
   - Identify key biological variables and their relationships
   - Specify input-output mappings and biological constraints
   - Define evaluation criteria with biological significance
   - Establish success metrics with biological validation

2. Analyze Key Challenges:
   - Identify biological and technical challenges
   - Assess data quality and biological variability
   - Evaluate computational complexity and scalability
   - Consider biological interpretability requirements
   - Address validation and reproducibility concerns

3. Formulate Research Questions:
   - Define primary research questions with biological focus
   - Identify key hypotheses to be tested
   - Specify biological mechanisms to be investigated
   - Outline experimental validation requirements
   - Establish biological significance criteria

4. Design Analysis Methods:
   - Propose computational approaches with biological relevance
   - Design validation strategies with biological context
   - Specify analysis pipelines with biological interpretation
   - Plan experimental validation with biological controls
   - Establish reproducibility standards with biological validation

Task Description:
{task_description}

Dataset Information:
{json.dumps(dataset_info, indent=2)}

Relevant Literature:
{papers_context}

Decision Support:
{json.dumps(decision_support, indent=2)}

Experimental Designs:
{json.dumps(experimental_designs, indent=2)}

Implementation Guides:
{json.dumps(implementation_guides, indent=2)}

Please provide a comprehensive investigation in the following JSON format:
{{
    "problem_definition": {{
        "formal_definition": {{
            "biological_context": "string",
            "input_output_mapping": "string",
            "biological_constraints": ["string"],
            "evaluation_criteria": ["string"],
            "success_metrics": ["string"]
        }},
        "key_variables": {{
            "biological": ["string"],
            "technical": ["string"],
            "relationships": ["string"],
            "constraints": ["string"],
            "validation_requirements": ["string"]
        }},
        "scope": {{
            "biological_scope": "string",
            "technical_scope": "string",
            "limitations": ["string"],
            "assumptions": ["string"],
            "biological_validation": "string"
        }}
    }},
    "key_challenges": {{
        "biological": {{
            "challenges": ["string"],
            "impact": "string",
            "mitigation": ["string"],
            "validation": "string",
            "biological_considerations": "string"
        }},
        "technical": {{
            "challenges": ["string"],
            "impact": "string",
            "mitigation": ["string"],
            "validation": "string",
            "implementation_requirements": "string"
        }},
        "data_quality": {{
            "issues": ["string"],
            "impact": "string",
            "mitigation": ["string"],
            "validation": "string",
            "biological_validation": "string"
        }},
        "computational": {{
            "challenges": ["string"],
            "impact": "string",
            "mitigation": ["string"],
            "validation": "string",
            "resource_requirements": "string"
        }},
        "interpretability": {{
            "requirements": ["string"],
            "challenges": ["string"],
            "solutions": ["string"],
            "validation": "string",
            "biological_validation": "string"
        }}
    }},
    "research_questions": {{
        "primary": {{
            "questions": ["string"],
            "hypotheses": ["string"],
            "biological_significance": "string",
            "validation_approach": "string",
            "expected_outcomes": ["string"]
        }},
        "secondary": {{
            "questions": ["string"],
            "hypotheses": ["string"],
            "biological_significance": "string",
            "validation_approach": "string",
            "expected_outcomes": ["string"]
        }},
        "biological_mechanisms": {{
            "mechanisms": ["string"],
            "investigation_approach": "string",
            "validation_methods": ["string"],
            "expected_insights": ["string"],
            "biological_validation": "string"
        }},
        "experimental_validation": {{
            "requirements": ["string"],
            "methods": ["string"],
            "controls": ["string"],
            "metrics": ["string"],
            "biological_validation": "string"
        }}
    }},
    "analysis_methods": {{
        "computational_approaches": {{
            "methods": ["string"],
            "rationale": "string",
            "implementation": "string",
            "validation": "string",
            "biological_validation": "string"
        }},
        "validation_strategies": {{
            "strategies": ["string"],
            "rationale": "string",
            "implementation": "string",
            "metrics": ["string"],
            "biological_validation": "string"
        }},
        "analysis_pipelines": {{
            "pipelines": ["string"],
            "components": ["string"],
            "workflow": "string",
            "validation": "string",
            "biological_interpretation": "string"
        }},
        "experimental_validation": {{
            "methods": ["string"],
            "controls": ["string"],
            "metrics": ["string"],
            "analysis": "string",
            "biological_validation": "string"
        }},
        "reproducibility": {{
            "standards": ["string"],
            "requirements": ["string"],
            "validation": "string",
            "documentation": "string",
            "biological_validation": "string"
        }}
    }}
}}""" 