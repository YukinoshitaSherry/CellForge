import os
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add current directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from .collaboration import CollaborationSystem
    from .data_structures import TaskAnalysisReport
except ImportError:
    from collaboration import CollaborationSystem
    from data_structures import TaskAnalysisReport

def initialize_knowledge_base(task_description: str, dataset_info: Dict[str, Any]) -> bool:
    """
    Initialize knowledge base with BFS-DFS search results
    This is called once at the beginning of Task Analysis
    
    Args:
        task_description: Description of the research task
        dataset_info: Information about the dataset
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from .rag import RAGSystem
        from .knowledge_base import knowledge_base
        
        print("Initializing knowledge base with BFS-DFS search...")
        
        # Initialize RAG system
        rag_system = RAGSystem()
        
        # Perform BFS-DFS search once
        print("Performing BFS-DFS search for papers...")
        papers = rag_system.search(task_description, limit=20)
        
        # Store papers in knowledge base (both databases)
        for paper in papers:
            # Store in CellForge (main database)
            knowledge_base.store_knowledge(
                knowledge_type="papers",
                content=paper,
                source="bfs_dfs_search",
                relevance_score=paper.get("score", 0.8),
                metadata=paper.get("metadata", {}),
                use_main_db=True  # CellForge
            )
            
            # Store in cellforge_tmp (temporary database)
            knowledge_base.store_knowledge(
                knowledge_type="papers",
                content=paper,
                source="bfs_dfs_search",
                relevance_score=paper.get("score", 0.8),
                metadata=paper.get("metadata", {}),
                use_main_db=False  # cellforge_tmp
            )
        
        # Get and store decision support information
        print("Retrieving decision support information...")
        decision_support = rag_system.get_decision_support(task_description, dataset_info)
        if decision_support:
            # Store in both databases
            knowledge_base.store_knowledge(
                knowledge_type="decision_support",
                content=decision_support,
                source="rag_system",
                relevance_score=0.9,
                metadata={"task": task_description},
                use_main_db=True
            )
            knowledge_base.store_knowledge(
                knowledge_type="decision_support",
                content=decision_support,
                source="rag_system",
                relevance_score=0.9,
                metadata={"task": task_description},
                use_main_db=False
            )
        
        # Get and store experimental designs
        print("Retrieving experimental designs...")
        experimental_designs = rag_system.search_experimental_designs(task_description)
        for design in experimental_designs:
            # Store in both databases
            knowledge_base.store_knowledge(
                knowledge_type="experimental_designs",
                content=design,
                source="rag_system",
                relevance_score=0.85,
                metadata={"task": task_description},
                use_main_db=True
            )
            knowledge_base.store_knowledge(
                knowledge_type="experimental_designs",
                content=design,
                source="rag_system",
                relevance_score=0.85,
                metadata={"task": task_description},
                use_main_db=False
            )
        
        # Get and store implementation guides
        print("Retrieving implementation guides...")
        implementation_guides = rag_system.search_implementation_guides(task_description)
        for guide in implementation_guides:
            # Store in both databases
            knowledge_base.store_knowledge(
                knowledge_type="implementation_guides",
                content=guide,
                source="rag_system",
                relevance_score=0.85,
                metadata={"task": task_description},
                use_main_db=True
            )
            knowledge_base.store_knowledge(
                knowledge_type="implementation_guides",
                content=guide,
                source="rag_system",
                relevance_score=0.85,
                metadata={"task": task_description},
                use_main_db=False
            )
        
        # Get and store evaluation frameworks
        print("Retrieving evaluation frameworks...")
        evaluation_frameworks = rag_system.search_evaluation_frameworks(task_description)
        for framework in evaluation_frameworks:
            # Store in both databases
            knowledge_base.store_knowledge(
                knowledge_type="evaluation_frameworks",
                content=framework,
                source="rag_system",
                relevance_score=0.85,
                metadata={"task": task_description},
                use_main_db=True
            )
            knowledge_base.store_knowledge(
                knowledge_type="evaluation_frameworks",
                content=framework,
                source="rag_system",
                relevance_score=0.85,
                metadata={"task": task_description},
                use_main_db=False
            )
        
        # Print summary
        summary = knowledge_base.get_knowledge_summary()
        print("✅ Knowledge base initialized successfully!")
        print(f"   Papers: {summary.get('papers', {}).get('count', 0)}")
        print(f"   Decision Support: {summary.get('decision_support', {}).get('count', 0)}")
        print(f"   Experimental Designs: {summary.get('experimental_designs', {}).get('count', 0)}")
        print(f"   Implementation Guides: {summary.get('implementation_guides', {}).get('count', 0)}")
        print(f"   Evaluation Frameworks: {summary.get('evaluation_frameworks', {}).get('count', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize knowledge base: {e}")
        return False

def run_task_analysis(task_description: str, dataset_info: dict = None):
    """
    Run complete Task Analysis workflow with knowledge base optimization
    
    Args:
        task_description: Description of the research task
        dataset_info: Information about the dataset
        
    Returns:
        TaskAnalysisReport with comprehensive analysis
    """
    try:
        print("Starting Task Analysis with optimized knowledge base...")
        
        # Initialize knowledge base with BFS-DFS search (only once)
        if not initialize_knowledge_base(task_description, dataset_info or {}):
            print("Knowledge base initialization failed, continuing with fallback...")
        
        # Initialize collaboration system (now uses knowledge base internally)
        from .collaboration import CollaborationSystem
        
        collaboration = CollaborationSystem()
        
        # Run analysis (agents will use knowledge base instead of repeated searches)
        print("Running collaborative analysis using knowledge base...")
        report = collaboration.run_analysis(
            task_description=task_description,
            dataset_info=dataset_info or {}
        )
        
        print("✅ Task Analysis completed successfully!")
        return report
        
    except Exception as e:
        print(f"❌ Task Analysis failed: {e}")
        return None

def main():
    """Main entry point"""
    
    # Example task description for CRISPR interference prediction
    task_description = """Develop a predictive model that accurately estimates gene expression profiles of individual K562 cells following CRISPR interference (CRISPRi), using the dataset from Norman et al. (2019, Science).

Task Definition:
- Input: Baseline gene expression profile of an unperturbed K562 cell and the identity of the target gene(s) for perturbation
- Output: Predicted gene expression profile after perturbation

Evaluation Scenarios:
1. Unseen Perturbations: Predict effects of gene perturbations not present during training
2. Unseen Cell Contexts: Predict responses in cells with gene expression profiles not observed during training

Evaluation Metrics:
- Mean Squared Error (MSE): Measures the average squared difference between predicted and observed gene expression.
- Pearson Correlation Coefficient (PCC): Quantifies linear correlation between predicted and observed profiles.
- R² (Coefficient of Determination): Represents the proportion of variance in the observed gene expression that can be explained by the predicted values.
- MSE for Differentially Expressed (DE) Genes (MSE_DE): Same as MSE but computed specifically for genes identified as differentially expressed.
- PCC for Differentially Expressed (DE) Genes (PCC_DE): Same as PCC but computed specifically for genes identified as differentially expressed.
- R² for Differentially Expressed (DE) Genes (R2_DE): Same as R² but computed specifically for genes identified as differentially expressed."""
    
    # Example dataset info
    dataset_info = {
        "dataset_name": "norman_2019_k562",
        "dataset_path": "cellforge/data/datasets/",
        "data_type": "scRNA-seq",
        "cell_line": "K562",
        "perturbation_type": "CRISPRi"
    }
    
    # Run the analysis
    run_task_analysis(task_description, dataset_info)

if __name__ == "__main__":
    main() 