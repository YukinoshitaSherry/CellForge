from typing import Dict, Any, Optional, Union
import json
import numpy as np
import os
import sys

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from graph_discussion import GraphDiscussion
    from refinement import RefinementFramework, RefinementConfig
    from experts import ExpertPool
except ImportError:
    # Fallback for relative imports
    from .graph_discussion import GraphDiscussion
    from .refinement import RefinementFramework, RefinementConfig
    from .experts import ExpertPool

class ResearchPlanGenerator:
    def __init__(self, task_analysis: Union[str, Dict[str, Any]], 
                 knowledge_base_vectors: Optional[np.ndarray] = None,
                 rag_retriever=None):
        """
        Initialize the research plan generator.
        
        Args:
            task_analysis: Either a path to JSON file or a dictionary containing task analysis
            knowledge_base_vectors: Optional knowledge base vectors for RAG
            rag_retriever: Optional RAG retriever for knowledge retrieval
        """
        self.task_analysis = self._load_task_analysis(task_analysis)
        self.knowledge_base = knowledge_base_vectors
        self.rag_retriever = rag_retriever
        self.expert_pool = ExpertPool()
        self.discussion = None
        self.refinement = None
        
    def _load_task_analysis(self, task_analysis: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Load task analysis from file or use provided dictionary."""
        if isinstance(task_analysis, str):
            with open(task_analysis, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif isinstance(task_analysis, dict):
            return task_analysis
        else:
            raise ValueError("task_analysis must be either a string (file path) or a dictionary")
            
    def generate_plan(self, task_type: str = None) -> Dict[str, Any]:
        """
        Generate a comprehensive research plan based on task analysis and knowledge base.
        
        Args:
            task_type: Type of task (gene_knockout, drug_perturbation, cytokine_stimulation)
        """
        # Determine task type if not provided
        if task_type is None:
            task_type = self._infer_task_type()
        
        # Initialize discussion framework with expert pool
        self.discussion = GraphDiscussion(
            expert_pool=self.expert_pool,
            task_analysis=self.task_analysis,
            knowledge_base=self.knowledge_base,
            rag_retriever=self.rag_retriever
        )
        
        # Run expert discussions
        discussion_results = self.discussion.run_discussion(
            task=self.task_analysis,
            task_type=task_type
        )
        
        # Initialize refinement framework
        config = RefinementConfig(
            seed=42,
            train_test_split=0.8,
            batch_size=64,
            learning_rate=3e-4,
            max_epochs=100,
            early_stopping_patience=15
        )
        
        self.refinement = RefinementFramework(config)
        self.refinement.integrate_discussion_results(discussion_results)
        
        # Generate final plan
        return {
            "markdown": self.refinement.generate_plan_markdown(),
            "json": self.refinement.final_plan,
            "mermaid": self.refinement.generate_mermaid_diagram(),
            "discussion_summary": discussion_results.get("summary", {}),
            "expert_contributions": discussion_results.get("expert_contributions", {})
        }
    
    def _infer_task_type(self) -> str:
        """Infer task type from task analysis content."""
        task_analysis = self.task_analysis
        
        # Check for task type indicators
        if "task_type" in task_analysis:
            return task_analysis["task_type"]
        
        # Infer from content
        content = str(task_analysis).lower()
        
        if any(keyword in content for keyword in ["gene", "knockout", "crispr", "sgrna"]):
            return "gene_knockout"
        elif any(keyword in content for keyword in ["drug", "compound", "chemical", "perturbation"]):
            return "drug_perturbation"
        elif any(keyword in content for keyword in ["cytokine", "stimulation", "ligand"]):
            return "cytokine_stimulation"
        else:
            # Default to gene knockout if unclear
            return "gene_knockout"
        
    def save_plan(self, output_dir: str):
        """Save the generated plan to files."""
        if not self.refinement:
            raise ValueError("Plan not generated yet. Call generate_plan() first.")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save markdown version
        self.refinement.save_plan(f"{output_dir}/research_plan.md", format="markdown")
        
        # Save JSON version
        self.refinement.save_plan(f"{output_dir}/research_plan.json", format="json")
        
        # Save mermaid diagram
        with open(f"{output_dir}/architecture.mmd", 'w', encoding='utf-8') as f:
            f.write(self.refinement.generate_mermaid_diagram())
        
        # Save discussion summary
        if hasattr(self.discussion, 'consensus_history'):
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.plot(self.discussion.consensus_history)
                plt.title('Discussion Consensus Progress')
                plt.xlabel('Round')
                plt.ylabel('Consensus Score')
                plt.savefig(f"{output_dir}/consensus_progress.png", dpi=300, bbox_inches='tight')
                plt.close()
            except ImportError:
                print("Warning: matplotlib not available, skipping consensus plot")

def generate_research_plan(task_analysis: Union[str, Dict[str, Any]], 
                         knowledge_base_vectors: Optional[np.ndarray] = None,
                         rag_retriever=None,
                         task_type: str = None,
                         output_dir: str = "results") -> Dict[str, Any]:
    """
    Generate a comprehensive research plan based on task analysis and optional knowledge base.
    
    Args:
        task_analysis: Path to the task analysis JSON file or dictionary
        knowledge_base_vectors: Optional knowledge base vectors for RAG
        rag_retriever: Optional RAG retriever for knowledge retrieval
        task_type: Type of task (gene_knockout, drug_perturbation, cytokine_stimulation)
        output_dir: Directory to save the output files
        
    Returns:
        Dict containing the generated plan in different formats
    """
    generator = ResearchPlanGenerator(
        task_analysis=task_analysis, 
        knowledge_base_vectors=knowledge_base_vectors,
        rag_retriever=rag_retriever
    )
    plan = generator.generate_plan(task_type=task_type)
    generator.save_plan(output_dir)
    return plan 