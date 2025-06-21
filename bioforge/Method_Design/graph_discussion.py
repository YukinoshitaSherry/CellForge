from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import networkx as nx
import numpy as np
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from experts import Expert, ExpertPool
except ImportError:
    
    from .experts import Expert, ExpertPool

@dataclass
class DiscussionNode:
    expert: Expert
    confidence: float
    proposal: Dict[str, Any]
    feedback: List[Dict[str, Any]]
    contribution_score: float = 0.0
    
class GraphDiscussion:
    def __init__(self, expert_pool: ExpertPool, 
                 task_analysis: Dict[str, Any] = None,
                 knowledge_base: Optional[np.ndarray] = None,
                 rag_retriever=None):
        self.expert_pool = expert_pool
        self.task_analysis = task_analysis
        self.knowledge_base = knowledge_base
        self.rag_retriever = rag_retriever
        self.graph = nx.Graph()
        self.nodes: Dict[str, DiscussionNode] = {}
        self.round = 0
        self.max_rounds = 10
        self.confidence_threshold = 0.8
        self.convergence_threshold = 0.1
        self.min_rounds = 3
        self.consensus_history = []
        self.discussion_nodes = []
        self.task = None
        self.task_type = None
        
    def initialize_discussion(self, task_analysis: Dict[str, Any], task_type: str):
        """Initialize the discussion with selected experts."""
        selected_experts = self.expert_pool.select_experts_for_task(task_type, task_analysis)
        
        for expert in selected_experts:
            node_id = f"{expert.name}_{self.round}"
            self.nodes[node_id] = DiscussionNode(
                expert=expert,
                confidence=0.5,
                proposal={},
                feedback=[],
                contribution_score=0.0
            )
            self.graph.add_node(node_id)
            
        for i, expert1 in enumerate(selected_experts):
            for expert2 in selected_experts[i+1:]:
                node1 = f"{expert1.name}_{self.round}"
                node2 = f"{expert2.name}_{self.round}"
                self.graph.add_edge(node1, node2)
                
        self.task_analysis = task_analysis
        self.task_type = task_type
        
    def run_discussion(self, task: Dict[str, Any], task_type: str = None, max_rounds: int = 10) -> Dict[str, Any]:
        """Run the expert discussion process."""
        self._initialize_discussion(task, task_type)
        
        for round_idx in range(max_rounds):
            print(f"\n=== Round {round_idx + 1} ===")
            
            
            initial_proposal = self._generate_initial_proposal()
            
            
            feedback = self._collect_feedback(initial_proposal)
            
            
            improvements = self._analyze_feedback(feedback)
            
            
            refined_proposal = self._refine_proposal(initial_proposal, improvements)
            
            
            self._update_confidences(feedback)
            self._update_contribution_scores(feedback)
            
            
            consensus_score = self._calculate_consensus_score()
            self.consensus_history.append(consensus_score)
            
            print(f"Consensus score: {consensus_score:.3f}")
            
            
            if self._check_termination(round_idx):
                print("Consensus reached, terminating discussion")
                break
                
        return self._generate_final_proposal()
        
    def _initialize_discussion(self, task: Dict[str, Any], task_type: str) -> None:
        """Initialize discussion nodes for all experts."""
        self.task = task
        self.task_type = task_type
        self.discussion_nodes = []
        self.current_round = 0
        self.consensus_reached = False
        
        
        selected_experts = self.expert_pool.select_experts_for_task(task_type, task)
        
        for expert in selected_experts:
            node = DiscussionNode(
                expert=expert,
                confidence=0.5,
                proposal=None,
                feedback=[],
                contribution_score=0.0
            )
            self.discussion_nodes.append(node)
            
    def _generate_initial_proposal(self) -> Dict[str, Any]:
        """Generate initial proposal using task analysis and RAG knowledge."""
        
        rag_knowledge = self._retrieve_rag_knowledge()
        
        proposal = {
            "preprocessing": self._generate_preprocessing_proposal(rag_knowledge),
            "feature_selection": self._generate_feature_selection_proposal(rag_knowledge),
            "batch_correction": self._generate_batch_correction_proposal(rag_knowledge),
            "biological_constraints": self._generate_biological_constraints(rag_knowledge),
            "cell_type_specificity": self._generate_cell_type_proposal(rag_knowledge),
            "pathway_integration": self._generate_pathway_proposal(rag_knowledge),
            "model_architecture": self._generate_architecture_proposal(rag_knowledge),
            "training_strategy": self._generate_training_proposal(rag_knowledge),
            "optimization_strategy": self._generate_optimization_proposal(rag_knowledge)
        }
        return proposal
    
    def _retrieve_rag_knowledge(self) -> Dict[str, Any]:
        """Retrieve relevant knowledge from RAG system."""
        if self.rag_retriever is None:
            return {}
        
        try:
            
            query_terms = self._extract_query_terms()
            
            
            retrieved_knowledge = {}
            for term in query_terms:
                results = self.rag_retriever.retrieve(term, top_k=5)
                retrieved_knowledge[term] = results
            
            return retrieved_knowledge
        except Exception as e:
            print(f"Warning: RAG retrieval failed: {e}")
            return {}
    
    def _extract_query_terms(self) -> List[str]:
        """Extract key terms from task analysis for RAG retrieval."""
        terms = []
        
        if self.task_analysis:
            
            if "task_type" in self.task_analysis:
                terms.append(self.task_analysis["task_type"])
            
            
            if "dataset" in self.task_analysis:
                dataset_info = self.task_analysis["dataset"]
                if "name" in dataset_info:
                    terms.append(dataset_info["name"])
                if "type" in dataset_info:
                    terms.append(dataset_info["type"])
            
            
            if "perturbations" in self.task_analysis:
                perturbations = self.task_analysis["perturbations"]
                for pert in perturbations:
                    if "type" in pert:
                        terms.append(pert["type"])
                    if "targets" in pert:
                        terms.extend(pert["targets"])
            
            
            if "cell_types" in self.task_analysis:
                terms.extend(self.task_analysis["cell_types"])
        
        
        if self.task_type:
            if self.task_type == "gene_knockout":
                terms.extend(["CRISPR", "gene knockout", "single cell RNA-seq"])
            elif self.task_type == "drug_perturbation":
                terms.extend(["drug response", "chemical perturbation", "dose response"])
            elif self.task_type == "cytokine_stimulation":
                terms.extend(["cytokine", "signaling", "immune response"])
        
        return list(set(terms))  
        
    def _collect_feedback(self, proposal: Dict[str, Any]) -> List[Dict[str, Any]]:
        feedback = []
        
        for node in self.discussion_nodes:
            expert_feedback = self._generate_feedback(node.expert, proposal)
            feedback.append({
                "expert": node.expert.name,
                "domain": node.expert.domain.value,
                "feedback": expert_feedback
            })
            node.feedback.append(expert_feedback)
            
        return feedback
        
    def _refine_proposal(self, proposal: Dict[str, Any], improvements: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        refined_proposal = proposal.copy()
        
        if "data_engineering" in improvements:
            de_improvements = improvements["data_engineering"]
            if "quality_control" in de_improvements:
                refined_proposal["preprocessing"]["quality_control"].update(
                    de_improvements["quality_control"]
                )
            if "normalization" in de_improvements:
                refined_proposal["preprocessing"]["normalization"].update(
                    de_improvements["normalization"]
                )
            if "batch_correction" in de_improvements:
                refined_proposal["batch_correction"].update(
                    de_improvements["batch_correction"]
                )
                
        if "single_cell_biology" in improvements:
            bio_improvements = improvements["single_cell_biology"]
            if "cell_type_specificity" in bio_improvements:
                refined_proposal["cell_type_specificity"].update(
                    bio_improvements["cell_type_specificity"]
                )
            if "pathway_constraints" in bio_improvements:
                refined_proposal["biological_constraints"]["pathway_constraints"].update(
                    bio_improvements["pathway_constraints"]
                )
            if "perturbation_responses" in bio_improvements:
                refined_proposal["biological_constraints"]["perturbation_responses"].update(
                    bio_improvements["perturbation_responses"]
                )
                
        if "deep_learning" in improvements:
            dl_improvements = improvements["deep_learning"]
            if "architecture" in dl_improvements:
                refined_proposal["model_architecture"].update(
                    dl_improvements["architecture"]
                )
            if "training" in dl_improvements:
                refined_proposal["training_strategy"].update(
                    dl_improvements["training"]
                )
            if "optimization" in dl_improvements:
                refined_proposal["optimization_strategy"].update(
                    dl_improvements["optimization"]
                )
                
        return refined_proposal
        
    def _update_confidences(self, feedback: List[Dict[str, Any]]) -> None:
        for node in self.discussion_nodes:
            feedback_quality = self._calculate_feedback_quality(node.feedback[-1])
            suggestion_impact = self._calculate_suggestion_impact(node.feedback[-1])
            
            confidence_update = (feedback_quality * 0.6 + suggestion_impact * 0.4) * 0.2
            node.confidence = min(1.0, node.confidence + confidence_update)
            
    def _calculate_feedback_quality(self, feedback: Dict[str, Any]) -> float:
        quality = 0.0
        
        if feedback["suggestions"]:
            quality += 0.4
        if feedback["weaknesses"]:
            quality += 0.3
        if feedback["strengths"]:
            quality += 0.3
            
        return quality
        
    def _update_contribution_scores(self, feedback: List[Dict[str, Any]]) -> None:
        for node in self.discussion_nodes:
            feedback_quality = self._calculate_feedback_quality(node.feedback[-1])
            suggestion_impact = self._calculate_suggestion_impact(node.feedback[-1])
            domain_relevance = self._calculate_domain_relevance(node.expert)
            
            contribution = (feedback_quality * 0.4 + 
                          suggestion_impact * 0.4 + 
                          domain_relevance * 0.2)
            
            node.contribution_score = min(1.0, node.contribution_score + contribution * 0.2)
            
    def _calculate_suggestion_impact(self, feedback: Dict[str, Any]) -> float:
        impact = 0.0
        if feedback["suggestions"]:
            impact += 0.4
        if feedback["weaknesses"]:
            impact += 0.3
        if feedback["strengths"]:
            impact += 0.3
        return impact
        
    def _calculate_domain_relevance(self, expert: Expert) -> float:
        task_domains = set(self.task_analysis.get("domains", []))
        expert_domains = set(expert.domain.value.split("_"))
        overlap = len(task_domains.intersection(expert_domains))
        return min(1.0, overlap / len(task_domains)) if task_domains else 0.5
        
    def _calculate_consensus_score(self) -> float:
        if not self.discussion_nodes:
            return 0.0
            
        confidences = [node.confidence for node in self.discussion_nodes]
        contributions = [node.contribution_score for node in self.discussion_nodes]
        
        weighted_confidences = [c * w for c, w in zip(confidences, contributions)]
        return sum(weighted_confidences) / len(weighted_confidences)
        
    def _check_termination(self, current_round: int) -> bool:
        if current_round < self.min_rounds:
            return False
            
        confidences = [node.confidence for node in self.discussion_nodes]
        if all(conf >= self.confidence_threshold for conf in confidences):
            if len(self.consensus_history) >= 2:
                score_change = abs(self.consensus_history[-1] - self.consensus_history[-2])
                if score_change < self.convergence_threshold:
                    self.consensus_reached = True
                    return True
                    
        return False
        
    def _generate_final_proposal(self) -> Dict[str, Any]:
        if not self.consensus_reached:
            print("Warning: Full consensus not reached")
        
        
        final_proposal = self._generate_initial_proposal()
        
        final_proposal["discussion_summary"] = {
            "rounds": self.current_round,
            "consensus_reached": self.consensus_reached,
            "final_consensus_score": self.consensus_history[-1] if self.consensus_history else 0.0,
            "expert_contributions": {
                node.expert.name: {
                    "confidence": node.confidence,
                    "contribution_score": node.contribution_score
                }
                for node in self.discussion_nodes
            }
        }
        
        return final_proposal

    def _generate_preprocessing_proposal(self, rag_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "quality_control": {
                "cell_filtering": {
                    "min_genes": 200,
                    "max_genes": 6000,
                    "min_counts": 1000,
                    "max_counts": 50000,
                    "max_mito_percent": 20
                },
                "gene_filtering": {
                    "min_cells": 3,
                    "min_counts": 10
                }
            },
            "normalization": {
                "method": "log1p",
                "target_sum": 10000,
                "regression_vars": ["total_counts", "pct_counts_mt"]
            },
            "batch_correction": {
                "method": "harmony",
                "parameters": {
                    "theta": 2,
                    "max_iterations": 20
                }
            }
        }
        
    def _generate_feature_selection_proposal(self, rag_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "highly_variable_genes": {
                "min_mean": 0.0125,
                "max_mean": 3,
                "min_disp": 0.5
            },
            "perturbation_targets": {
                "include_all": True,
                "min_cells": 3
            },
            "dimensionality_reduction": {
                "method": "pca",
                "n_components": 512,
                "svd_solver": "arpack"
            }
        }
        
    def _generate_batch_correction_proposal(self, rag_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "method": "harmony",
            "parameters": {
                "theta": 2,
                "max_iterations": 20,
                "batch_key": "batch"
            },
            "evaluation": {
                "metrics": ["kBET", "LISI"],
                "visualization": ["umap", "tsne"]
            }
        }
        
    def _generate_biological_constraints(self, rag_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "cell_type_specificity": {
                "markers": {
                    "immune_cells": ["CD3D", "CD4", "CD8A"],
                    "cancer_cells": ["EPCAM", "KRT8", "KRT18"]
                },
                "perturbation_responses": {
                    "cytokine": ["IL2", "IFNg", "TNFa"],
                    "drug": ["Cyclosporin", "Rapamycin"]
                }
            },
            "pathway_constraints": {
                "signaling": ["MAPK", "PI3K_AKT", "JAK_STAT"],
                "stress_response": ["heat_shock", "oxidative"],
                "cell_cycle": ["G1/S", "G2/M"]
            }
        }
        
    def _generate_cell_type_proposal(self, rag_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "embedding_dimension": 128,
            "interaction": "element_wise_multiplication",
            "validation": {
                "markers": ["TP53", "BCL2", "CDKN1A"],
                "pathways": ["apoptosis", "cell_cycle", "stress_response"]
            }
        }
        
    def _generate_pathway_proposal(self, rag_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "databases": ["KEGG", "Reactome", "GO", "MSigDB"],
            "analysis": {
                "method": "gsea",
                "parameters": {
                    "min_size": 5,
                    "max_size": 500
                }
            },
            "visualization": {
                "methods": ["heatmap", "network", "enrichment_plot"]
            }
        }
        
    def _generate_architecture_proposal(self, rag_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "encoder": {
                "type": "transformer",
                "layers": 4,
                "heads": 8,
                "dimension": 512,
                "activation": "gelu"
            },
            "perturbation_encoder": {
                "type": "mlp",
                "layers": 3,
                "dimensions": [256, 128, 64],
                "activation": "swish"
            },
            "cross_attention": {
                "heads": 8,
                "dimension": 512,
                "dropout": 0.1
            },
            "decoder": {
                "type": "mlp",
                "layers": 3,
                "dimensions": [256, 512, 1024],
                "activation": "swish"
            }
        }
        
    def _generate_training_proposal(self, rag_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "optimizer": {
                "type": "adamw",
                "learning_rate": 3e-4,
                "weight_decay": 0.01
            },
            "scheduler": {
                "type": "cosine_annealing",
                "T_0": 10,
                "eta_min": 1e-6
            },
            "training": {
                "batch_size": 64,
                "gradient_clip": 1.0,
                "early_stopping": {
                    "patience": 15,
                    "min_delta": 1e-4
                }
            }
        }
        
    def _generate_optimization_proposal(self, rag_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "loss_functions": {
                "reconstruction": "mse",
                "perturbation": "bce",
                "biological": "pathway_consistency"
            },
            "regularization": {
                "dropout": 0.1,
                "weight_decay": 0.01,
                "gradient_clip": 1.0
            },
            "mixed_precision": True,
            "gradient_checkpointing": True
        }
        
    def _analyze_feedback(self, feedback: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        improvements = {}
        
        for f in feedback:
            expert_domain = f["domain"]
            expert_feedback = f["feedback"]
            
            if expert_domain not in improvements:
                improvements[expert_domain] = {}
                
            
            if expert_domain == "data_engineering":
                improvements[expert_domain].update(
                    self._analyze_data_engineering_feedback(expert_feedback)
                )
            elif expert_domain == "single_cell_biology":
                improvements[expert_domain].update(
                    self._analyze_biology_feedback(expert_feedback)
                )
            elif expert_domain == "deep_learning":
                improvements[expert_domain].update(
                    self._analyze_deep_learning_feedback(expert_feedback)
                )
                
        return improvements
        
    def _generate_feedback(self, expert: Expert, proposal: Dict[str, Any]) -> Dict[str, Any]:
        feedback = {
            "strengths": [],
            "weaknesses": [],
            "suggestions": []
        }
        
        if expert.domain.value == "data_engineering":
            feedback.update(self._generate_data_engineering_feedback(proposal))
        elif expert.domain.value == "single_cell_biology":
            feedback.update(self._generate_biology_feedback(proposal))
        elif expert.domain.value == "deep_learning":
            feedback.update(self._generate_deep_learning_feedback(proposal))
            
        return feedback

    def _analyze_data_engineering_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        improvements = {
            "quality_control": {},
            "normalization": {},
            "batch_correction": {}
        }
        
        if "quality_control" in feedback:
            qc_feedback = feedback["quality_control"]
            if "cell_filtering" in qc_feedback:
                improvements["quality_control"]["cell_filtering"] = {
                    "min_genes": max(200, qc_feedback["cell_filtering"].get("min_genes", 200)),
                    "max_genes": min(6000, qc_feedback["cell_filtering"].get("max_genes", 6000)),
                    "min_counts": max(1000, qc_feedback["cell_filtering"].get("min_counts", 1000)),
                    "max_counts": min(50000, qc_feedback["cell_filtering"].get("max_counts", 50000))
                }
                
        if "normalization" in feedback:
            norm_feedback = feedback["normalization"]
            improvements["normalization"] = {
                "method": norm_feedback.get("method", "log1p"),
                "target_sum": norm_feedback.get("target_sum", 10000),
                "regression_vars": norm_feedback.get("regression_vars", ["total_counts", "pct_counts_mt"])
            }
            
        if "batch_correction" in feedback:
            bc_feedback = feedback["batch_correction"]
            improvements["batch_correction"] = {
                "method": bc_feedback.get("method", "harmony"),
                "parameters": {
                    "theta": bc_feedback.get("theta", 2),
                    "max_iterations": bc_feedback.get("max_iterations", 20)
                }
            }
            
        return improvements
        
    def _analyze_biology_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        improvements = {
            "cell_type_specificity": {},
            "pathway_constraints": {},
            "perturbation_responses": {}
        }
        
        if "cell_type_specificity" in feedback:
            ct_feedback = feedback["cell_type_specificity"]
            improvements["cell_type_specificity"] = {
                "markers": ct_feedback.get("markers", {}),
                "perturbation_responses": ct_feedback.get("perturbation_responses", {})
            }
            
        if "pathway_constraints" in feedback:
            pathway_feedback = feedback["pathway_constraints"]
            improvements["pathway_constraints"] = {
                "signaling": pathway_feedback.get("signaling", []),
                "stress_response": pathway_feedback.get("stress_response", []),
                "cell_cycle": pathway_feedback.get("cell_cycle", [])
            }
            
        if "perturbation_responses" in feedback:
            pert_feedback = feedback["perturbation_responses"]
            improvements["perturbation_responses"] = {
                "cytokine": pert_feedback.get("cytokine", []),
                "drug": pert_feedback.get("drug", [])
            }
            
        return improvements
        
    def _analyze_deep_learning_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        improvements = {
            "architecture": {},
            "training": {},
            "optimization": {}
        }
        
        if "architecture" in feedback:
            arch_feedback = feedback["architecture"]
            improvements["architecture"] = {
                "encoder": {
                    "type": arch_feedback.get("encoder_type", "transformer"),
                    "layers": arch_feedback.get("encoder_layers", 4),
                    "heads": arch_feedback.get("encoder_heads", 8),
                    "dimension": arch_feedback.get("encoder_dimension", 512)
                },
                "perturbation_encoder": {
                    "type": arch_feedback.get("pert_encoder_type", "mlp"),
                    "layers": arch_feedback.get("pert_encoder_layers", 3),
                    "dimensions": arch_feedback.get("pert_encoder_dims", [256, 128, 64])
                }
            }
            
        if "training" in feedback:
            train_feedback = feedback["training"]
            improvements["training"] = {
                "optimizer": {
                    "type": train_feedback.get("optimizer_type", "adamw"),
                    "learning_rate": train_feedback.get("learning_rate", 3e-4),
                    "weight_decay": train_feedback.get("weight_decay", 0.01)
                },
                "scheduler": {
                    "type": train_feedback.get("scheduler_type", "cosine_annealing"),
                    "T_0": train_feedback.get("T_0", 10),
                    "eta_min": train_feedback.get("eta_min", 1e-6)
                }
            }
            
        if "optimization" in feedback:
            opt_feedback = feedback["optimization"]
            improvements["optimization"] = {
                "loss_functions": opt_feedback.get("loss_functions", {}),
                "regularization": opt_feedback.get("regularization", {}),
                "mixed_precision": opt_feedback.get("mixed_precision", True)
            }
            
        return improvements
        
    def _generate_data_engineering_feedback(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        feedback = {
            "strengths": [],
            "weaknesses": [],
            "suggestions": []
        }
        
        if "quality_control" in proposal:
            qc = proposal["quality_control"]
            if "cell_filtering" in qc:
                cf = qc["cell_filtering"]
                if cf["min_genes"] < 200:
                    feedback["weaknesses"].append("minimum gene number is too low")
                    feedback["suggestions"].append("suggest to increase the minimum gene number to 200 or more")
                if cf["max_genes"] > 6000:
                    feedback["weaknesses"].append("maximum gene number is too high")
                    feedback["suggestions"].append("suggest to limit the maximum gene number to 6000 or less")
                    
        if "normalization" in proposal:
            norm = proposal["normalization"]
            if norm["method"] == "log1p":
                feedback["strengths"].append("log1p normalization is suitable for single-cell data")
            else:
                feedback["suggestions"].append("suggest to use log1p normalization")
                
        if "batch_correction" in proposal:
            bc = proposal["batch_correction"]
            if bc["method"] == "harmony":
                feedback["strengths"].append("harmony is suitable for batch correction")
            else:
                feedback["suggestions"].append("suggest to use harmony for batch correction")
                
        return feedback
        
    def _generate_biology_feedback(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        feedback = {
            "strengths": [],
            "weaknesses": [],
            "suggestions": []
        }
        
        if "cell_type_specificity" in proposal:
            ct = proposal["cell_type_specificity"]
            if "markers" in ct:
                feedback["strengths"].append("includes key cell type markers")
            if "perturbation_responses" in ct:
                feedback["strengths"].append("considers perturbation response genes")
                
        if "pathway_constraints" in proposal:
            pathway = proposal["pathway_constraints"]
            if "signaling" in pathway:
                feedback["strengths"].append("includes important signaling pathways")
            if "stress_response" in pathway:
                feedback["strengths"].append("considers stress response pathways")
                
        return feedback
        
    def _generate_deep_learning_feedback(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        feedback = {
            "strengths": [],
            "weaknesses": [],
            "suggestions": []
        }
        
        if "architecture" in proposal:
            arch = proposal["architecture"]
            if arch["encoder"]["type"] == "transformer":
                feedback["strengths"].append("transformer encoder is suitable for sequence data")
            if arch["perturbation_encoder"]["type"] == "mlp":
                feedback["strengths"].append("MLP encoder is suitable for perturbation information")
                
        if "training" in proposal:
            train = proposal["training"]
            if train["optimizer"]["type"] == "adamw":
                feedback["strengths"].append("AdamW optimizer is suitable for training")
            if train["scheduler"]["type"] == "cosine_annealing":
                feedback["strengths"].append("cosine annealing scheduler is suitable for learning rate scheduling")
                
        if "optimization" in proposal:
            opt = proposal["optimization"]
            if "mixed_precision" in opt and opt["mixed_precision"]:
                feedback["strengths"].append("mixed precision training is suitable for training")
            if "gradient_checkpointing" in opt and opt["gradient_checkpointing"]:
                feedback["strengths"].append("gradient checkpointing is suitable for saving memory")
                
        return feedback
