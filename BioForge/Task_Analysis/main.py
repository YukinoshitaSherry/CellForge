import os
import json
import sys
from datetime import datetime
from pathlib import Path

# Add current directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from collaboration import CollaborationSystem
from data_structures import TaskAnalysisReport

def run_task_analysis(task_description: str, dataset_info: dict = None):
    """
    Main function to run the complete task analysis pipeline
    
    Args:
        task_description: The task description to analyze
        dataset_info: Dataset info dictionary (optional)
    """
    
    # Initialize collaboration system
    print("Initializing collaboration system...")
    collaboration_system = CollaborationSystem()
    
    # Use provided dataset info or create empty dict
    if dataset_info is None:
        dataset_info = {}
    
    # Run the analysis
    print("Starting task analysis...")
    print(f"Task: {task_description}")
    print(f"Dataset: {dataset_info.get('dataset_name', 'Not specified')}")
    
    try:
        # Execute the analysis
        report = collaboration_system.run_analysis(task_description, dataset_info)
        
        # Create results directory if it doesn't exist
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_filename = f"task_analysis_report_{timestamp}.json"
        json_path = results_dir / json_filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2, default=str)
        
        print(f"JSON report saved to: {json_path}")
        
        # Save Markdown report
        md_filename = f"task_analysis_report_{timestamp}.md"
        md_path = results_dir / md_filename
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report.to_markdown())
        
        print(f"Markdown report saved to: {md_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Task: {task_description}")
        print(f"Dataset: {dataset_info.get('dataset_name', 'Not specified')}")
        print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Reports saved to: {results_dir}")
        print("="*50)
        
        return report
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

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
        "dataset_path": "BioForge/data/datasets/",
        "data_type": "scRNA-seq",
        "cell_line": "K562",
        "perturbation_type": "CRISPRi"
    }
    
    # Run the analysis
    run_task_analysis(task_description, dataset_info)

if __name__ == "__main__":
    main() 