from typing import Dict, Any, Optional, Union
import json
import numpy as np
import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
    print(f"Loaded .env from: {env_path}")
except ImportError:
    print("Warning: python-dotenv not installed, using system environment variables")

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

        from pathlib import Path
        from datetime import datetime
        project_root = Path(__file__).parent.parent.parent  # scAgents根目录

        if output_dir == "results" or "results" in output_dir:
            output_path = project_root / "cellforge" / "data" / "results"
        else:
            output_path = Path(output_dir)

        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"research_plan_{timestamp}"
        
        print(f"Saving research plan to: {output_path}")
        print(f"Base filename: {base_filename}")
        
        # Save markdown version
        self.refinement.save_plan(str(output_path / f"{base_filename}.md"), format="markdown")
        
        # Save JSON version
        self.refinement.save_plan(str(output_path / f"{base_filename}.json"), format="json")
        
        # Save mermaid diagram
        try:
            with open(output_path / f"{base_filename}.mmd", 'w', encoding='utf-8') as f:
                f.write(self.refinement.generate_mermaid_diagram())
        except Exception as e:
            print(f"Warning: Could not save mermaid diagram: {e}")
        
        # Save discussion summary
        if hasattr(self.discussion, 'consensus_history'):
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.plot(self.discussion.consensus_history)
                plt.title('Discussion Consensus Progress')
                plt.xlabel('Round')
                plt.ylabel('Consensus Score')
                plt.savefig(str(output_path / f"{base_filename}_consensus.png"), dpi=300, bbox_inches='tight')
                plt.close()
            except ImportError:
                print("Warning: matplotlib not available, skipping consensus plot")
            except Exception as e:
                print(f"Warning: Could not save consensus plot: {e}")
        return base_filename

def generate_research_plan(task_analysis: Union[str, Dict[str, Any]], 
                         knowledge_base_vectors: Optional[np.ndarray] = None,
                         rag_retriever=None,
                         task_type: str = None,
                         output_dir: str = "results",
                         auto_generate_code: bool = True) -> Dict[str, Any]:
    """
    Generate a comprehensive research plan based on task analysis and optional knowledge base.
    
    Args:
        task_analysis: Path to the task analysis JSON file or dictionary
        knowledge_base_vectors: Optional knowledge base vectors for RAG
        rag_retriever: Optional RAG retriever for knowledge retrieval
        task_type: Type of task (gene_knockout, drug_perturbation, cytokine_stimulation)
        output_dir: Directory to save the output files
        auto_generate_code: Whether to automatically generate code after plan generation
        
    Returns:
        Dict containing the generated plan in different formats and file information
    """
    generator = ResearchPlanGenerator(
        task_analysis=task_analysis, 
        knowledge_base_vectors=knowledge_base_vectors,
        rag_retriever=rag_retriever
    )
    plan = generator.generate_plan(task_type=task_type)
    base_filename = generator.save_plan(output_dir)

    plan['generated_files'] = {
        'base_filename': base_filename,
        'output_directory': output_dir,
        'files': [
            f"{base_filename}.md",
            f"{base_filename}.json", 
            f"{base_filename}.mmd",
            f"{base_filename}_consensus.png"
        ]
    }

    if auto_generate_code:
        try:
            print("\n=== Starting Automatic Code Generation ===")
            print("Initializing OpenHands code generator...")
            
            # Import code generation module
            try:
                from ..Code_Generation import generate_code_from_plan
            except ImportError:
                from Code_Generation import generate_code_from_plan
            
            # Generate code using OpenHands
            code_file_path = generate_code_from_plan(plan, output_dir)
            
            if code_file_path:
                print(f"✅ Code generated successfully: {code_file_path}")
                plan['generated_files']['files'].append("result.py")
                plan['code_generation'] = {
                    'status': 'success',
                    'file_path': code_file_path,
                    'generated_at': datetime.now().isoformat()
                }
            else:
                print("❌ Code generation failed")
                plan['code_generation'] = {
                    'status': 'failed',
                    'error': 'Failed to generate code using OpenHands',
                    'generated_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"❌ Error in code generation: {e}")
            plan['code_generation'] = {
                'status': 'error',
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    return plan 