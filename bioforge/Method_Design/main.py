#!/usr/bin/env python3
"""
Method Design Module - Complete Research Plan Generator
"""

import json
import numpy as np
import os
import sys
from typing import Dict, Any, List, Optional

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from __init__ import generate_research_plan

class RAGRetriever:
    """RAG knowledge retriever for biological research"""
    
    def __init__(self, knowledge_base: Optional[Dict[str, Any]] = None):
        self.knowledge_base = knowledge_base or self._load_default_knowledge()
    
    def _load_default_knowledge(self) -> Dict[str, Any]:
        """Load default knowledge base"""
        return {
            "CRISPR": {
                "description": "CRISPR-Cas9 gene editing technology",
                "best_practices": ["High-quality sgRNA design", "Proper controls", "Validation"],
                "data_processing": ["Guide detection", "Quality control", "Normalization"]
            },
            "single_cell": {
                "description": "Single-cell RNA sequencing analysis",
                "preprocessing": ["Quality control", "Normalization", "Feature selection"],
                "analysis": ["Dimensionality reduction", "Clustering", "Differential expression"]
            },
            "deep_learning": {
                "description": "Deep learning for biological data",
                "architectures": ["Transformer", "Graph Neural Networks", "Autoencoders"],
                "training": ["Transfer learning", "Multi-task learning", "Curriculum learning"]
            },
            "perturbation": {
                "description": "Perturbation analysis methods",
                "types": ["Gene knockout", "Drug treatment", "Cytokine stimulation"],
                "analysis": ["Differential expression", "Pathway enrichment", "Network analysis"]
            },
            "pathway": {
                "description": "Biological pathway analysis",
                "databases": ["KEGG", "Reactome", "GO", "MSigDB"],
                "methods": ["GSEA", "ORA", "Pathway enrichment"]
            }
        }
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge"""
        results = []
        query_lower = query.lower()
        
        for key, content in self.knowledge_base.items():
            relevance = 0.0
            
            # Check keyword matching
            if query_lower in key.lower():
                relevance += 0.8
            if query_lower in str(content).lower():
                relevance += 0.6
            
            if relevance > 0:
                results.append({
                    "content": content,
                    "source": key,
                    "relevance_score": min(1.0, relevance)
                })
        
        # Sort by relevance and return top_k
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]

def load_task_analysis(file_path: str) -> Dict[str, Any]:
    """Load task analysis from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Task analysis file not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {file_path}")

def create_sample_task_analysis() -> Dict[str, Any]:
    """Create sample task analysis"""
    return {
        "task_type": "gene_knockout",
        "dataset": {
            "name": "Norman_2019",
            "type": "single_cell_RNA_seq",
            "description": "CRISPR perturbation dataset with gene knockouts"
        },
        "perturbations": [
            {
                "type": "gene_knockout",
                "targets": ["ATF6", "XBP1", "IRE1"],
                "description": "UPR pathway genes"
            }
        ],
        "cell_types": ["K562", "A549", "MCF7"],
        "objectives": [
            "Predict gene expression changes after perturbation",
            "Understand UPR pathway dynamics",
            "Generalize to unseen perturbations"
        ],
        "constraints": [
            "Limited training data",
            "Need for biological interpretability",
            "Computational efficiency requirements"
        ],
        "evaluation_metrics": [
            "MSE",
            "Pearson correlation",
            "Biological pathway enrichment"
        ]
    }

def create_task_analysis_interactive() -> Dict[str, Any]:
    """Create task analysis interactively"""
    print("=== Create Task Analysis ===\n")
    
    task_analysis = {}
    
    # Task type
    print("Select task type:")
    print("1. gene_knockout")
    print("2. drug_perturbation")
    print("3. cytokine_stimulation")
    
    while True:
        choice = input("Enter choice (1-3): ").strip()
        if choice == "1":
            task_analysis["task_type"] = "gene_knockout"
            break
        elif choice == "2":
            task_analysis["task_type"] = "drug_perturbation"
            break
        elif choice == "3":
            task_analysis["task_type"] = "cytokine_stimulation"
            break
        else:
            print("Invalid choice, please try again")
    
    # Dataset information
    print("\nDataset information:")
    task_analysis["dataset"] = {
        "name": input("Dataset name: ").strip() or "Unknown",
        "type": input("Data type (e.g., single_cell_RNA_seq): ").strip() or "single_cell_RNA_seq",
        "description": input("Dataset description (optional): ").strip() or ""
    }
    
    # Perturbation information
    print("\nPerturbation information:")
    perturbation_type = input("Perturbation type: ").strip() or task_analysis["task_type"]
    targets = input("Target genes/molecules (comma-separated): ").strip()
    targets_list = [t.strip() for t in targets.split(",") if t.strip()]
    
    task_analysis["perturbations"] = [{
        "type": perturbation_type,
        "targets": targets_list,
        "description": input("Perturbation description (optional): ").strip() or ""
    }]
    
    # Cell types
    cell_types = input("Cell types (comma-separated): ").strip()
    task_analysis["cell_types"] = [ct.strip() for ct in cell_types.split(",") if ct.strip()]
    
    # Objectives
    print("\nResearch objectives (one per line, empty line to finish):")
    objectives = []
    while True:
        obj = input().strip()
        if not obj:
            break
        objectives.append(obj)
    task_analysis["objectives"] = objectives
    
    return task_analysis

def main():
    """Main function"""
    print("=== Method Design Module ===")
    print("Multi-Expert Research Plan Generator\n")
    
    # Get task analysis
    print("Select task analysis source:")
    print("1. Load from file")
    print("2. Interactive creation")
    print("3. Use sample data")
    
    while True:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            file_path = input("Enter task analysis file path: ").strip()
            try:
                task_analysis = load_task_analysis(file_path)
                print("✅ Successfully loaded task analysis file")
                break
            except Exception as e:
                print(f"❌ Loading failed: {e}")
                continue
        
        elif choice == "2":
            task_analysis = create_task_analysis_interactive()
            print("✅ Successfully created task analysis")
            break
        
        elif choice == "3":
            task_analysis = create_sample_task_analysis()
            print("✅ Using sample data")
            break
        
        else:
            print("Invalid choice, please try again")
    
    # Create RAG retriever
    print("\nInitializing RAG knowledge retriever...")
    rag_retriever = RAGRetriever()
    
    # Generate research plan
    print("\nGenerating research plan...")
    try:
        output_dir = input("Output directory (default: results): ").strip() or "results"
        
        plan = generate_research_plan(
            task_analysis=task_analysis,
            rag_retriever=rag_retriever,
            task_type=task_analysis.get("task_type"),
            output_dir=output_dir
        )
        
        print("\n✅ Research plan generated successfully!")
        print(f"Output directory: {output_dir}")
        
        # Show summary
        if 'discussion_summary' in plan:
            summary = plan['discussion_summary']
            print(f"Discussion rounds: {summary.get('rounds', 'N/A')}")
            print(f"Consensus reached: {summary.get('consensus_reached', 'N/A')}")
        
        if 'expert_contributions' in plan:
            experts = plan['expert_contributions']
            print(f"Participating experts: {len(experts)}")
            
            # Show expert contributions
            print("\nExpert contributions:")
            for expert_name, contribution in list(experts.items())[:5]:  # Show top 5
                confidence = contribution.get('confidence', 0)
                print(f"  - {expert_name}: confidence {confidence:.2f}")
        
        print(f"\nGenerated files:")
        print(f"  - {output_dir}/research_plan.md (Research plan)")
        print(f"  - {output_dir}/research_plan.json (Detailed data)")
        print(f"  - {output_dir}/architecture.mmd (Architecture diagram)")
        print(f"  - {output_dir}/consensus_progress.png (Consensus progress)")
        
        print("\n=== Complete ===")
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check:")
        print("1. All dependencies are installed")
        print("2. Task analysis format is correct")
        print("3. Output directory has write permissions")

if __name__ == "__main__":
    main() 