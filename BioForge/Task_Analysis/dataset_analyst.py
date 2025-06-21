from typing import Dict, Any, List
from datetime import datetime
import os
import json

from .data_structures import AnalysisResult
from .rag import RAGSystem
from .llm import LLMInterface

class DatasetAnalyst:
    """Expert agent for analyzing dataset characteristics and quality"""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        
        self.llm = LLMInterface()
    
    def analyze_dataset(self, task_description: str, dataset_info: Dict[str, Any],
                       retrieved_papers: List[Dict[str, Any]]) -> AnalysisResult:
        """
        Analyze dataset characteristics and quality
        
        Args:
            task_description: Description of the research task
            dataset_info: Dictionary containing dataset metadata
            retrieved_papers: List of relevant papers from vector database
            
        Returns:
            AnalysisResult with dataset analysis
        """
        # 获取RAG搜索结果和决策支持信息
        rag_results = self.rag_system.search(task_description, dataset_info)
        
        # 获取专门的决策支持信息
        decision_support = self.rag_system.get_decision_support(task_description, dataset_info)
        
        # 获取实验设计信息
        experimental_designs = self.rag_system.search_experimental_designs(task_description)
        
        # 合并所有论文信息
        all_papers = retrieved_papers + [
            {
                "title": result.title,
                "abstract": result.content,
                "metadata": result.metadata,
                "decision_support": result.metadata.get("decision_support", {})
            }
            for result in rag_results.get("papers", [])
        ]
        
        # Format prompt with task information and retrieved papers
        prompt = self._format_prompt_with_decision_support(
            task_description, dataset_info, all_papers, decision_support, experimental_designs
        )
        
        # Run analysis (implementation depends on your LLM backend)
        analysis_content = self._run_llm(prompt)
        
        return AnalysisResult(
            content=analysis_content,
            confidence_score=1.0,  # 临时设置，后续会由其他模块计算
            timestamp=datetime.now(),
            metadata={
                "agent": "Dataset Analyst",
                "rag_results": {
                    "papers_count": len(rag_results.get("papers", [])),
                    "decision_support_available": bool(decision_support),
                    "experimental_designs_count": len(experimental_designs)
                },
                "decision_support": decision_support
            }
        )
    
    def _format_prompt(self, task_description: str, dataset_info: Dict[str, Any],
                      retrieved_papers: List[Dict[str, Any]]) -> str:
        """Format prompt for dataset analysis"""
        papers_context = "\n".join([
            f"- {paper['title']}: {paper['abstract'][:200]}..."
            for paper in retrieved_papers[:5]
        ])
        
        return f"""You are an expert in single-cell dataset analysis, specializing in evaluating single-cell perturbation data. Your task is to provide a comprehensive analysis of dataset characteristics and quality, focusing on experimental design, data properties, preprocessing recommendations, and quality assessment.

1. Analyze Experimental Design:
   - Evaluate experimental setup and biological conditions
   - Assess sample size and biological replicates
   - Review perturbation design and controls
   - Analyze biological variability and batch effects
   - Consider technical and biological validation

2. Assess Data Properties:
   - Analyze data distribution and biological patterns
   - Evaluate feature space and biological relevance
   - Assess data quality and technical artifacts
   - Identify biological and technical noise
   - Consider data sparsity and missing values

3. Provide Preprocessing Recommendations:
   - Suggest normalization strategies for biological data
   - Recommend quality control procedures
   - Propose feature selection methods
   - Design batch effect correction approaches
   - Plan biological validation steps

4. Conduct Quality Assessment:
   - Evaluate data completeness and coverage
   - Assess biological relevance and interpretability
   - Analyze technical quality and reproducibility
   - Consider biological validation requirements
   - Identify potential limitations and biases

Task Description:
{task_description}

Dataset Information:
{json.dumps(dataset_info, indent=2)}

Relevant Literature:
{papers_context}

Please provide a comprehensive analysis in the following JSON format:
{{
    "experimental_design": {{
        "setup": {{
            "biological_conditions": ["string"],
            "experimental_protocol": "string",
            "controls": ["string"],
            "replicates": "string",
            "biological_validation": "string"
        }},
        "perturbation": {{
            "design": "string",
            "types": ["string"],
            "doses": ["string"],
            "timepoints": ["string"],
            "biological_controls": ["string"]
        }},
        "samples": {{
            "size": "string",
            "distribution": "string",
            "characteristics": ["string"],
            "biological_variability": "string",
            "technical_variability": "string"
        }},
        "validation": {{
            "biological": {{
                "methods": ["string"],
                "controls": ["string"],
                "metrics": ["string"],
                "results": "string",
                "interpretation": "string"
            }},
            "technical": {{
                "methods": ["string"],
                "controls": ["string"],
                "metrics": ["string"],
                "results": "string",
                "interpretation": "string"
            }}
        }}
    }},
    "data_properties": {{
        "distribution": {{
            "biological": {{
                "patterns": ["string"],
                "variability": "string",
                "outliers": ["string"],
                "interpretation": "string",
                "biological_significance": "string"
            }},
            "technical": {{
                "patterns": ["string"],
                "variability": "string",
                "artifacts": ["string"],
                "interpretation": "string",
                "mitigation": ["string"]
            }}
        }},
        "features": {{
            "biological": {{
                "types": ["string"],
                "relevance": "string",
                "selection": ["string"],
                "interpretation": "string",
                "validation": "string"
            }},
            "technical": {{
                "types": ["string"],
                "quality": "string",
                "selection": ["string"],
                "interpretation": "string",
                "validation": "string"
            }}
        }},
        "quality": {{
            "biological": {{
                "metrics": ["string"],
                "assessment": "string",
                "issues": ["string"],
                "mitigation": ["string"],
                "validation": "string"
            }},
            "technical": {{
                "metrics": ["string"],
                "assessment": "string",
                "issues": ["string"],
                "mitigation": ["string"],
                "validation": "string"
            }}
        }},
        "noise": {{
            "biological": {{
                "sources": ["string"],
                "impact": "string",
                "assessment": "string",
                "mitigation": ["string"],
                "validation": "string"
            }},
            "technical": {{
                "sources": ["string"],
                "impact": "string",
                "assessment": "string",
                "mitigation": ["string"],
                "validation": "string"
            }}
        }}
    }},
    "preprocessing": {{
        "normalization": {{
            "strategies": {{
                "biological": ["string"],
                "technical": ["string"],
                "rationale": "string",
                "implementation": "string",
                "validation": "string"
            }},
            "parameters": {{
                "biological": ["string"],
                "technical": ["string"],
                "optimization": "string",
                "validation": "string",
                "biological_validation": "string"
            }}
        }},
        "quality_control": {{
            "procedures": {{
                "biological": ["string"],
                "technical": ["string"],
                "rationale": "string",
                "implementation": "string",
                "validation": "string"
            }},
            "thresholds": {{
                "biological": ["string"],
                "technical": ["string"],
                "rationale": "string",
                "implementation": "string",
                "validation": "string"
            }}
        }},
        "feature_selection": {{
            "methods": {{
                "biological": ["string"],
                "technical": ["string"],
                "rationale": "string",
                "implementation": "string",
                "validation": "string"
            }},
            "criteria": {{
                "biological": ["string"],
                "technical": ["string"],
                "rationale": "string",
                "implementation": "string",
                "validation": "string"
            }}
        }},
        "batch_correction": {{
            "methods": {{
                "biological": ["string"],
                "technical": ["string"],
                "rationale": "string",
                "implementation": "string",
                "validation": "string"
            }},
            "parameters": {{
                "biological": ["string"],
                "technical": ["string"],
                "optimization": "string",
                "validation": "string",
                "biological_validation": "string"
            }}
        }}
    }},
    "quality_assessment": {{
        "completeness": {{
            "biological": {{
                "coverage": "string",
                "gaps": ["string"],
                "impact": "string",
                "mitigation": ["string"],
                "validation": "string"
            }},
            "technical": {{
                "coverage": "string",
                "gaps": ["string"],
                "impact": "string",
                "mitigation": ["string"],
                "validation": "string"
            }}
        }},
        "relevance": {{
            "biological": {{
                "significance": "string",
                "interpretability": "string",
                "validation": "string",
                "limitations": ["string"],
                "improvements": ["string"]
            }},
            "technical": {{
                "significance": "string",
                "interpretability": "string",
                "validation": "string",
                "limitations": ["string"],
                "improvements": ["string"]
            }}
        }},
        "reproducibility": {{
            "biological": {{
                "assessment": "string",
                "metrics": ["string"],
                "validation": "string",
                "limitations": ["string"],
                "improvements": ["string"]
            }},
            "technical": {{
                "assessment": "string",
                "metrics": ["string"],
                "validation": "string",
                "limitations": ["string"],
                "improvements": ["string"]
            }}
        }},
        "limitations": {{
            "biological": {{
                "issues": ["string"],
                "impact": "string",
                "mitigation": ["string"],
                "validation": "string",
                "improvements": ["string"]
            }},
            "technical": {{
                "issues": ["string"],
                "impact": "string",
                "mitigation": ["string"],
                "validation": "string",
                "improvements": ["string"]
            }}
        }}
    }}
}}"""

    def _format_prompt_with_decision_support(self, task_description: str, dataset_info: Dict[str, Any],
                                           papers: List[Dict[str, Any]], decision_support: Dict[str, Any],
                                           experimental_designs: List[Dict[str, Any]]) -> str:
        """Format prompt for dataset analysis with decision support"""
        papers_context = "\n".join([
            f"- {paper['title']}: {paper['abstract'][:200]}..."
            for paper in papers[:5]
        ])
        
        # 格式化决策支持信息
        decision_context = ""
        if decision_support:
            decision_context = f"""
DECISION SUPPORT INFORMATION:
Data Preparation: {json.dumps(decision_support.get('data_preparation', {}), indent=2)}
Implementation Plan: {json.dumps(decision_support.get('implementation_plan', {}), indent=2)}
Risk Assessment: {json.dumps(decision_support.get('risk_assessment', {}), indent=2)}
"""
        
        # 格式化实验设计信息
        design_context = ""
        if experimental_designs:
            design_context = f"""
EXPERIMENTAL DESIGNS:
{chr(10).join([f"- {design['title']}: {design['content'][:200]}..." for design in experimental_designs[:3]])}
"""
        
        return f"""You are an expert in single-cell dataset analysis, specializing in evaluating single-cell perturbation data. Your task is to provide a comprehensive analysis of dataset characteristics and quality, focusing on experimental design, data properties, preprocessing recommendations, and quality assessment.

{decision_context}

{design_context}

1. Analyze Experimental Design:
   - Evaluate experimental setup and biological conditions
   - Assess sample size and biological replicates
   - Review perturbation design and controls
   - Analyze biological variability and batch effects
   - Consider technical and biological validation

2. Assess Data Properties:
   - Analyze data distribution and biological patterns
   - Evaluate feature space and biological relevance
   - Assess data quality and technical artifacts
   - Identify biological and technical noise
   - Consider data sparsity and missing values

3. Provide Preprocessing Recommendations:
   - Suggest normalization strategies for biological data
   - Recommend quality control procedures
   - Propose feature selection methods
   - Design batch effect correction approaches
   - Plan biological validation steps

4. Conduct Quality Assessment:
   - Evaluate data completeness and coverage
   - Assess biological relevance and interpretability
   - Analyze technical quality and reproducibility
   - Consider biological validation requirements
   - Identify potential limitations and biases

Task Description:
{task_description}

Dataset Information:
{json.dumps(dataset_info, indent=2)}

Relevant Literature:
{papers_context}

Please provide a comprehensive analysis in the following JSON format, incorporating decision support information:
{{
    "experimental_design": {{
        "setup": {{
            "biological_conditions": ["string"],
            "experimental_protocol": "string",
            "controls": ["string"],
            "replicates": "string",
            "biological_validation": "string",
            "decision_support_alignment": "string"
        }},
        "perturbation": {{
            "design": "string",
            "types": ["string"],
            "doses": ["string"],
            "timepoints": ["string"],
            "biological_controls": ["string"],
            "decision_support_considerations": "string"
        }},
        "samples": {{
            "size": "string",
            "distribution": "string",
            "characteristics": ["string"],
            "biological_variability": "string",
            "technical_variability": "string",
            "decision_support_requirements": "string"
        }},
        "validation": {{
            "biological": {{
                "methods": ["string"],
                "controls": ["string"],
                "metrics": ["string"],
                "results": "string",
                "interpretation": "string",
                "decision_support_validation": "string"
            }},
            "technical": {{
                "methods": ["string"],
                "controls": ["string"],
                "metrics": ["string"],
                "results": "string",
                "interpretation": "string",
                "decision_support_validation": "string"
            }}
        }}
    }},
    "data_properties": {{
        "distribution": {{
            "biological": {{
                "patterns": ["string"],
                "variability": "string",
                "outliers": ["string"],
                "interpretation": "string",
                "biological_significance": "string",
                "decision_support_implications": "string"
            }},
            "technical": {{
                "patterns": ["string"],
                "variability": "string",
                "artifacts": ["string"],
                "interpretation": "string",
                "mitigation": ["string"],
                "decision_support_considerations": "string"
            }}
        }},
        "features": {{
            "biological": {{
                "types": ["string"],
                "relevance": "string",
                "selection": ["string"],
                "interpretation": "string",
                "validation": "string",
                "decision_support_features": "string"
            }},
            "technical": {{
                "types": ["string"],
                "quality": "string",
                "selection": ["string"],
                "interpretation": "string",
                "validation": "string",
                "decision_support_technical": "string"
            }}
        }},
        "quality": {{
            "biological": {{
                "metrics": ["string"],
                "assessment": "string",
                "issues": ["string"],
                "mitigation": ["string"],
                "validation": "string",
                "decision_support_quality": "string"
            }},
            "technical": {{
                "metrics": ["string"],
                "assessment": "string",
                "issues": ["string"],
                "mitigation": ["string"],
                "validation": "string",
                "decision_support_technical_quality": "string"
            }}
        }},
        "noise": {{
            "biological": {{
                "sources": ["string"],
                "impact": "string",
                "assessment": "string",
                "mitigation": ["string"],
                "validation": "string",
                "decision_support_noise": "string"
            }},
            "technical": {{
                "sources": ["string"],
                "impact": "string",
                "assessment": "string",
                "mitigation": ["string"],
                "validation": "string",
                "decision_support_technical_noise": "string"
            }}
        }}
    }},
    "preprocessing": {{
        "normalization": {{
            "strategies": {{
                "recommendations": ["string"],
                "rationale": "string",
                "implementation": "string",
                "validation": "string",
                "biological_validation": "string",
                "decision_support_normalization": "string"
            }},
            "quality_control": {{
                "procedures": ["string"],
                "criteria": ["string"],
                "implementation": "string",
                "validation": "string",
                "biological_validation": "string",
                "decision_support_qc": "string"
            }},
            "feature_selection": {{
                "methods": ["string"],
                "criteria": ["string"],
                "implementation": "string",
                "validation": "string",
                "biological_validation": "string",
                "decision_support_features": "string"
            }},
            "batch_correction": {{
                "methods": ["string"],
                "criteria": ["string"],
                "implementation": "string",
                "validation": "string",
                "biological_validation": "string",
                "decision_support_batch": "string"
            }}
        }},
        "validation": {{
            "biological": {{
                "methods": ["string"],
                "criteria": ["string"],
                "implementation": "string",
                "validation": "string",
                "biological_validation": "string",
                "decision_support_validation": "string"
            }},
            "technical": {{
                "methods": ["string"],
                "criteria": ["string"],
                "implementation": "string",
                "validation": "string",
                "technical_validation": "string",
                "decision_support_technical_validation": "string"
            }}
        }}
    }},
    "quality_assessment": {{
        "completeness": {{
            "coverage": "string",
            "missing_data": "string",
            "assessment": "string",
            "mitigation": ["string"],
            "validation": "string",
            "decision_support_completeness": "string"
        }},
        "relevance": {{
            "biological": "string",
            "technical": "string",
            "assessment": "string",
            "validation": "string",
            "biological_validation": "string",
            "decision_support_relevance": "string"
        }},
        "reproducibility": {{
            "technical": "string",
            "biological": "string",
            "assessment": "string",
            "validation": "string",
            "biological_validation": "string",
            "decision_support_reproducibility": "string"
        }},
        "limitations": {{
            "biological": ["string"],
            "technical": ["string"],
            "assessment": "string",
            "mitigation": ["string"],
            "validation": "string",
            "decision_support_limitations": "string"
        }}
    }}
}}"""

    def _run_llm(self, prompt: str) -> Dict[str, Any]:
        system_prompt = "You are an expert in single-cell dataset analysis. Provide your response in valid JSON format."
        
        try:
            # 使用统一的LLMInterface
            response = self.llm.generate(prompt, system_prompt)
            return response
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}") 