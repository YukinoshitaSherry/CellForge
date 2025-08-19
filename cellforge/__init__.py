"""
cellforge: Open-Ended Autonomous Design of Computational Methods for Single-Cell Omics via Multi-Agent Collaboration

A cutting-edge end-to-end multi-agent framework that revolutionizes single-cell data analysis 
through intelligent task decomposition, automated method design, and collaborative problem-solving.
"""

__version__ = "0.1.0"
__author__ = "cellforge Team"
__email__ = "cellforge@example.com"

# Import main components
try:
    from .Task_Analysis import *
    from .Method_Design import *
    from .Code_Generation import *
    from .RAG import *
    from .llm import LLMInterface
except ImportError:
    # Allow partial imports for development
    pass

# Main workflow function
def run_end_to_end_workflow(task_description: str, dataset_info: dict = None):
    """
    Run the complete end-to-end workflow from task analysis to code generation.
    
    Args:
        task_description: Description of the analysis task
        dataset_info: Optional dataset information dictionary
        
    Returns:
        bool: True if workflow completed successfully, False otherwise
    """
    from .end_to_end_workflow import EndToEndWorkflow
    
    workflow = EndToEndWorkflow()
    return workflow.run_complete_workflow(task_description, dataset_info)

# Convenience imports
__all__ = [
    "run_end_to_end_workflow",
    "LLMInterface",
    "Task_Analysis",
    "Method_Design", 
    "Code_Generation",
    "RAG"
] 