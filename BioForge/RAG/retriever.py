from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from .search import HybridSearcher, SearchResult
from .utils import TextProcessor

class HybridRetriever:
    """
    Hybrid retriever that combines local and online search capabilities
    """
    def __init__(self, qdrant_url: str = "localhost", qdrant_port: int = 6333):
        """
        Initialize hybrid retriever
        
        Args:
            qdrant_url: Qdrant server URL
            qdrant_port: Qdrant server port
        """
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.qdrant_client = QdrantClient(url=qdrant_url, port=qdrant_port)
        self.hybrid_searcher = HybridSearcher(qdrant_url, qdrant_port)
        self.text_processor = TextProcessor()
        self._load_single_cell_terms()
        
    def search(self, task_description: str, dataset_info: Dict[str, Any]) -> Dict[str, List[SearchResult]]:
        """
        Search for relevant papers and code with enhanced decision support
        
        Args:
            task_description: Description of the research task
            dataset_info: Information about the dataset
            
        Returns:
            Dictionary containing search results with decision support
        """
        keywords = self._extract_keywords(task_description, dataset_info)
        results = self.hybrid_searcher.search(keywords)
        
        for source, search_results in results.items():
            for result in search_results:
                result.metadata = {
                    "perturbation_type": self._identify_perturbation_type(result.content),
                    "technology": self._identify_technology(result.content),
                    "analysis_method": self._identify_analysis_method(result.content),
                    "cell_type": self._identify_cell_type(result.content),
                    "perturbation_effect": self._identify_perturbation_effect(result.content),
                    "model_type": self._identify_model_type(result.content),
                    "evaluation_metric": self._identify_evaluation_metric(result.content),
                    "dataset_characteristics": self._identify_dataset_characteristics(result.content),
                    "decision_support": self._extract_decision_support(result)
                }
        
        results["decision_recommendations"] = self._generate_decision_recommendations(
            task_description, dataset_info, results
        )
        
        return results
    
    def _extract_keywords(self, task_description: str, dataset_info: Dict[str, Any]) -> List[str]:
        """
        Extract keywords from task description and dataset info
        
        Args:
            task_description: Description of the research task
            dataset_info: Information about the dataset
            
        Returns:
            List of keywords
        """
        text = f"{task_description} {dataset_info.get('name', '')} {dataset_info.get('modality', '')}"
        technical_terms = self.text_processor.extract_technical_terms(text)
        biological_terms = self.text_processor.extract_biological_terms(text)
        keywords = self.text_processor.extract_keywords(text)
        single_cell_terms = []
        
        for category, terms in self.single_cell_terms.items():
            single_cell_terms.extend(terms)
        
        all_terms = list(set(technical_terms + biological_terms + keywords + single_cell_terms))
        return all_terms
    
    def _identify_perturbation_type(self, text: str) -> List[str]:
        return [term for term in self.single_cell_terms["perturbation_types"] 
                if term.lower() in text.lower()]
    
    def _identify_technology(self, text: str) -> List[str]:
        return [term for term in self.single_cell_terms["technologies"] 
                if term.lower() in text.lower()]
    
    def _identify_analysis_method(self, text: str) -> List[str]:
        return [term for term in self.single_cell_terms["analysis_methods"] 
                if term.lower() in text.lower()]
    
    def _identify_cell_type(self, text: str) -> List[str]:
        return [term for term in self.single_cell_terms["cell_types"] 
                if term.lower() in text.lower()]
    
    def _identify_perturbation_effect(self, text: str) -> List[str]:
        return [term for term in self.single_cell_terms["perturbation_effects"] 
                if term.lower() in text.lower()]
    
    def _identify_model_type(self, text: str) -> List[str]:
        return [term for term in self.single_cell_terms["model_types"] 
                if term.lower() in text.lower()]
    
    def _identify_evaluation_metric(self, text: str) -> List[str]:
        return [term for term in self.single_cell_terms["evaluation_metrics"] 
                if term.lower() in text.lower()]
    
    def _identify_dataset_characteristics(self, text: str) -> List[str]:
        return [term for term in self.single_cell_terms["dataset_characteristics"] 
                if term.lower() in text.lower()]
    
    def _extract_decision_support(self, result: SearchResult) -> Dict[str, Any]:
        decision_support = result.metadata.get("decision_support", {})
        
        if not decision_support:
            decision_support = self._extract_decision_info_from_content(result.content)
        
        return decision_support
    
    def _extract_decision_info_from_content(self, content: str) -> Dict[str, Any]:
        decision_info = {
            "model_recommendations": [],
            "evaluation_metrics": [],
            "data_requirements": [],
            "implementation_complexity": "unknown",
            "biological_interpretability": "unknown",
            "confidence_score": 0.0
        }
        
        for model_type in self.single_cell_terms["ml_models"]:
            if model_type.lower() in content.lower():
                decision_info["model_recommendations"].append(model_type)
        
        for metric in self.single_cell_terms["evaluation_metrics"]:
            if metric.lower() in content.lower():
                decision_info["evaluation_metrics"].append(metric)
        
        for characteristic in self.single_cell_terms["dataset_characteristics"]:
            if characteristic.lower() in content.lower():
                decision_info["data_requirements"].append(characteristic)
        
        confidence_factors = [
            len(decision_info["model_recommendations"]) > 0,
            len(decision_info["evaluation_metrics"]) > 0,
            len(decision_info["data_requirements"]) > 0
        ]
        decision_info["confidence_score"] = sum(confidence_factors) / len(confidence_factors)
        
        return decision_info
    
    def _generate_decision_recommendations(self, task_description: str, 
                                         dataset_info: Dict[str, Any], 
                                         results: Dict[str, List[SearchResult]]) -> List[Dict[str, Any]]:
        recommendations = []
        
        all_decision_support = []
        for source, search_results in results.items():
            if source != "decision_recommendations":
                for result in search_results:
                    if hasattr(result, 'metadata') and result.metadata:
                        decision_support = result.metadata.get("decision_support", {})
                        if decision_support:
                            all_decision_support.append(decision_support)
        
        model_recommendations = self._aggregate_model_recommendations(all_decision_support)
        if model_recommendations:
            recommendations.append({
                "type": "model_recommendation",
                "title": "Recommended Models",
                "content": model_recommendations,
                "confidence": 0.8
            })
        
        metric_recommendations = self._aggregate_metric_recommendations(all_decision_support)
        if metric_recommendations:
            recommendations.append({
                "type": "evaluation_recommendation",
                "title": "Recommended Evaluation Metrics",
                "content": metric_recommendations,
                "confidence": 0.7
            })
        
        data_recommendations = self._aggregate_data_recommendations(all_decision_support, dataset_info)
        if data_recommendations:
            recommendations.append({
                "type": "data_recommendation",
                "title": "Data Requirements",
                "content": data_recommendations,
                "confidence": 0.6
            })
        
        return recommendations
    
    def _aggregate_model_recommendations(self, decision_support_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        model_counts = {}
        complexity_scores = {"very_high": 4, "high": 3, "moderate": 2, "low": 1}
        
        for support in decision_support_list:
            for model in support.get("model_recommendations", []):
                model_counts[model] = model_counts.get(model, 0) + 1
            
            complexity = support.get("implementation_complexity", "unknown")
            if complexity in complexity_scores:
                model_counts["complexity_score"] = model_counts.get("complexity_score", 0) + complexity_scores[complexity]
        
        sorted_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
        top_models = [model for model, count in sorted_models[:5] if model != "complexity_score"]
        
        return {
            "recommended_models": top_models,
            "complexity_level": "high" if model_counts.get("complexity_score", 0) > 10 else "moderate"
        }
    
    def _aggregate_metric_recommendations(self, decision_support_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        metric_counts = {}
        
        for support in decision_support_list:
            for metric in support.get("evaluation_metrics", []):
                metric_counts[metric] = metric_counts.get(metric, 0) + 1
        
        sorted_metrics = sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)
        top_metrics = [metric for metric, count in sorted_metrics[:5]]
        
        return {
            "primary_metrics": top_metrics[:3],
            "secondary_metrics": top_metrics[3:],
            "metric_count": len(top_metrics)
        }
    
    def _aggregate_data_recommendations(self, decision_support_list: List[Dict[str, Any]], 
                                      dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        data_requirements = set()
        
        for support in decision_support_list:
            for requirement in support.get("data_requirements", []):
                data_requirements.add(requirement)
        
        dataset_characteristics = dataset_info.get("characteristics", [])
        missing_requirements = list(data_requirements - set(dataset_characteristics))
        
        return {
            "required_data_types": list(data_requirements),
            "missing_requirements": missing_requirements,
            "dataset_compatibility": len(missing_requirements) == 0
        }
    
    def _load_single_cell_terms(self) -> None:
        self.single_cell_terms = {
            "perturbation_types": [
                "gene_knockout", "gene_knockdown", "CRISPR", "RNAi",
                "drug_treatment", "small_molecule", "compound",
                "overexpression", "transfection", "transduction",
                "perturbation", "intervention", "modification"
            ],
            "technologies": [
                "scRNA-seq", "single-cell RNA sequencing", "10x Genomics",
                "Drop-seq", "Smart-seq2", "CEL-seq2", "sci-RNA-seq",
                "Perturb-seq", "CROP-seq", "Mosaic-seq",
                "single-cell ATAC-seq", "single-cell proteomics",
                "spatial transcriptomics", "multi-omics"
            ],
            "analysis_methods": [
                "differential_expression", "trajectory_analysis",
                "pseudotime", "cell_type_annotation", "clustering",
                "dimensionality_reduction", "batch_correction",
                "integration", "regulatory_network",
                "gene_regulatory_network", "pathway_analysis",
                "enrichment_analysis", "cell_state_transition"
            ],
            "cell_types": [
                "T_cell", "B_cell", "macrophage", "dendritic_cell",
                "stem_cell", "neuron", "fibroblast", "epithelial_cell",
                "cancer_cell", "immune_cell", "progenitor_cell",
                "endothelial_cell", "mesenchymal_cell"
            ],
            "perturbation_effects": [
                "cell_state_change", "differentiation", "proliferation",
                "apoptosis", "cell_cycle", "signaling_pathway",
                "transcriptional_regulation", "epigenetic_modification",
                "metabolic_change", "immune_response", "stress_response",
                "cell_fate_decision", "cellular_plasticity"
            ],
            "model_types": [
                "deep_learning", "neural_network", "transformer",
                "graph_neural_network", "variational_autoencoder",
                "generative_adversarial_network", "recurrent_neural_network",
                "convolutional_neural_network", "attention_mechanism",
                "self_supervised_learning", "transfer_learning"
            ],
            "evaluation_metrics": [
                "accuracy", "precision", "recall", "f1_score",
                "auc_roc", "mean_squared_error", "r2_score",
                "silhouette_score", "adjusted_rand_index",
                "normalized_mutual_information", "pearson_correlation",
                "spearman_correlation", "jaccard_similarity"
            ],
            "dataset_characteristics": [
                "high_dimensional", "sparse", "noisy", "imbalanced",
                "batch_effect", "dropout", "technical_variation",
                "biological_variation", "temporal_dynamics",
                "spatial_heterogeneity", "cell_type_heterogeneity",
                "perturbation_heterogeneity"
            ],
            "ml_models": [
                # Foundation Models
                "transformer", "bert", "gpt", "t5", "roberta",
                "encoder", "decoder", "attention_mechanism", "embedding",
                
                # Deep Learning Architectures
                "convolutional_neural_network", "recurrent_neural_network",
                "lstm", "gru", "resnet", "gnn", "graph_neural_network",
                "autoencoder", "vae", "gan", "diffusion_model",
                
                # Traditional ML Models
                "random_forest", "svm", "xgboost", "lightgbm",
                "gradient_boosting", "decision_tree", "knn",
                
                # Single-cell Foundation Models
                "scgpt", "geneformer", "scbert", "scgpt2", "scgpt3",
                "scgpt4", "scgpt5", "scgpt6", "scgpt7", "scgpt8",
                
                # Perturbation Prediction Models
                "chemcpa", "gears", "perturbnet", "perturbseq",
                "perturbation_transformer", "perturbation_bert",
                "perturbation_vae", "perturbation_gan",
                
                # Gene Expression Models
                "gene_expression_transformer", "gene_expression_bert",
                "gene_expression_vae", "gene_expression_gan",
                "gene_expression_autoencoder", "gene_expression_lstm",
                
                # Cell Type Models
                "cell_type_transformer", "cell_type_bert",
                "cell_type_vae", "cell_type_gan",
                "cell_type_autoencoder", "cell_type_lstm",
                
                # Trajectory Models
                "trajectory_transformer", "trajectory_bert",
                "trajectory_vae", "trajectory_gan",
                "trajectory_autoencoder", "trajectory_lstm",
                
                # Integration Models
                "scvi", "scarches", "scanvi", "totalvi", "peakvi",
                "integration_transformer", "integration_bert",
                "integration_vae", "integration_gan",
                "integration_autoencoder", "integration_lstm",
                
                # Benchmark Models
                "benchmark_transformer", "benchmark_bert",
                "benchmark_vae", "benchmark_gan",
                "benchmark_autoencoder", "benchmark_lstm",
                
                # Specialized Models
                "drug_response", "cell_fate", "cell_cycle",
                "cell_state", "cell_identity", "cell_communication",
                "cell_signaling", "cell_metabolism", "cell_development",
                "cell_differentiation", "cell_proliferation",
                "cell_apoptosis", "cell_migration", "cell_invasion",
                
                # Multi-modal Models
                "multi_modal_transformer", "multi_modal_bert",
                "multi_modal_vae", "multi_modal_gan",
                "multi_modal_autoencoder", "multi_modal_lstm",
                
                # Transfer Learning Models
                "transfer_learning_transformer", "transfer_learning_bert",
                "transfer_learning_vae", "transfer_learning_gan",
                "transfer_learning_autoencoder", "transfer_learning_lstm",
                
                # Few-shot Learning Models
                "few_shot_transformer", "few_shot_bert",
                "few_shot_vae", "few_shot_gan",
                "few_shot_autoencoder", "few_shot_lstm",
                
                # Zero-shot Learning Models
                "zero_shot_transformer", "zero_shot_bert",
                "zero_shot_vae", "zero_shot_gan",
                "zero_shot_autoencoder", "zero_shot_lstm"
            ],
            "ml_techniques": [
                "transfer_learning", "few_shot_learning", "zero_shot_learning",
                "self_supervised_learning", "semi_supervised_learning",
                "active_learning", "meta_learning", "federated_learning",
                "reinforcement_learning", "curriculum_learning",
                "knowledge_distillation", "model_pruning", "quantization"
            ],
            "ml_frameworks": [
                "pytorch", "tensorflow", "keras", "jax", "mxnet",
                "scikit_learn", "huggingface", "fastai", "lightning",
                "onnx", "tensorrt", "torchscript", "tflite"
            ],
            "ml_metrics": [
                "accuracy", "precision", "recall", "f1_score",
                "auc_roc", "mean_squared_error", "r2_score",
                "cross_entropy", "perplexity", "bleu_score",
                "rouge_score", "bert_score", "inception_score",
                "fid_score", "perplexity"
            ],
            "ml_optimization": [
                "adam", "sgd", "adamw", "rmsprop", "momentum",
                "learning_rate_scheduling", "gradient_clipping",
                "weight_decay", "dropout", "batch_normalization",
                "layer_normalization", "early_stopping",
                "model_checkpointing", "mixed_precision"
            ]
        } 