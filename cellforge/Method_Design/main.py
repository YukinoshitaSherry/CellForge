#!/usr/bin/env python3
"""
Method Design Module - Complete Research Plan Generator
"""

import json
import numpy as np
import os
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # 明确指定.env文件路径
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
    print(f"Loaded .env from: {env_path}")
except ImportError:
    print("Warning: python-dotenv not installed, using system environment variables")

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from __init__ import generate_research_plan

class RAGRetriever:
    """RAG knowledge retriever using MCP knowledgebase for Qdrant access"""
    
    def __init__(self, knowledge_base: Optional[Dict[str, Any]] = None):
        self.knowledge_base = knowledge_base or self._load_default_knowledge()
        self.mcp_client = None
        self._initialize_mcp_client()
    
    def _initialize_mcp_client(self):
        """Initialize MCP client for knowledgebase access"""
        try:
            # Try to import and initialize MCP client
            try:
                from ..Task_Analysis.rag import RAGSystem
            except ImportError:
                from Task_Analysis.rag import RAGSystem
            # Use the same RAG system as Task Analysis
            self.rag_system = RAGSystem()
            print("✅ MCP knowledgebase client initialized successfully")
        except Exception as e:
            print(f"⚠️ MCP client initialization failed: {e}")
            print("Falling back to default knowledge base")
            self.rag_system = None
    
    def _load_default_knowledge(self) -> Dict[str, Any]:
        """Load default knowledge base as fallback"""
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
        """Retrieve relevant knowledge from Qdrant via MCP knowledgebase"""
        if self.rag_system:
            try:
                # Use the same BFS-DFS search strategy as Task Analysis
                results = self.rag_system.search(query, top_k=top_k)
                
                # Convert results to expected format
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "content": result.get("content", result),
                        "source": result.get("source", "qdrant"),
                        "relevance_score": result.get("score", 0.8)
                    })
                
                return formatted_results
                
            except Exception as e:
                print(f"⚠️ MCP knowledgebase retrieval failed: {e}")
                print("Falling back to default knowledge base")
        
        # Fallback to default knowledge base
        return self._retrieve_from_default_kb(query, top_k)
    
    def _retrieve_from_default_kb(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve from default knowledge base"""
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

def load_task_analysis(file_path: str = None, plan_id: str = None, latest: bool = True) -> Dict[str, Any]:
    """
    Load task analysis from file with enhanced path resolution
    
    Args:
        file_path: Direct file path (absolute or relative)
        plan_id: Specific plan ID to find
        latest: If True and no plan_id/file_path, return latest plan
        
    Returns:
        Task analysis data
    """
    try:
        # If no file_path provided, try to find latest task analysis from results directory
        if not file_path:
            # First try to find task analysis from results directory
            project_root = Path(__file__).parent.parent.parent  # scAgents root
            results_dir = project_root / "cellforge" / "data" / "results"
            
            if latest:
                # Look for task analysis files in results directory
                task_analysis_files = []
                
                # Look for analysis_report.json (task analysis output)
                analysis_report = results_dir / "analysis_report.json"
                if analysis_report.exists():
                    task_analysis_files.append(analysis_report)
                
                # Look for other potential task analysis files
                for pattern in ["*analysis*.json", "*task*.json", "*report*.json"]:
                    task_analysis_files.extend(results_dir.glob(pattern))
                
                if task_analysis_files:
                    # Get the most recent file
                    latest_file = max(task_analysis_files, key=lambda x: x.stat().st_mtime)
                    file_path = str(latest_file)
                    print(f"Found latest task analysis file: {latest_file.name}")
                else:
                    # Fallback to plans directory
                    try:
                        try:
                            from ..Task_Analysis.plan_storage import plan_storage
                        except ImportError:
                            from Task_Analysis.plan_storage import plan_storage
                        file_path = plan_storage.get_plan_file_path(plan_id=plan_id, latest=latest)
                        if not file_path:
                            raise FileNotFoundError("No research plan found")
                    except ImportError:
                        # Fallback to default location
                        plans_dir = project_root / "cellforge" / "data" / "plans"
                        
                        if latest:
                            files = list(plans_dir.glob("research_plan_*.json"))
                            if files:
                                file_path = str(max(files, key=lambda x: x.stat().st_mtime))
                            else:
                                raise FileNotFoundError("No research plan files found")
                        else:
                            raise FileNotFoundError("No file path or plan ID provided")
            else:
                raise FileNotFoundError("No file path or plan ID provided")
        
        # Resolve path (handle relative paths)
        target_path = Path(file_path)
        if not target_path.is_absolute():
            # Try relative to current directory
            current_dir = Path.cwd()
            target_path = current_dir / target_path
            
            # If not found, try relative to project root
            if not target_path.exists():
                project_root = Path(__file__).parent.parent.parent
                target_path = project_root / file_path
        
        # Load the file
        with open(target_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert task analysis format to expected format
        if isinstance(data, dict):
            # Check if it's an analysis report (task analysis output)
            if 'task_requirements' in data and 'dataset_characteristics' in data:
                # Convert analysis report to task analysis format
                task_analysis = {
                    "task_type": "drug_perturbation",  # Default based on L1000 data
                    "dataset": {
                        "name": "L1000_Connectivity_Map",
                        "type": "bulk_RNA_seq",
                        "description": data.get('dataset_characteristics', {}).get('source_protocol', 'L1000 Connectivity Map')
                    },
                    "perturbations": [
                        {
                            "type": "drug_perturbation",
                            "targets": ["Small molecule compounds"],
                            "description": "Drug perturbation analysis using L1000 data"
                        }
                    ],
                    "cell_types": data.get('dataset_characteristics', {}).get('composition', {}).get('cell_lines', ['A549', 'MCF7', 'PC3']),
                    "objectives": [
                        data.get('task_requirements', {}).get('core_task_definition', 'Predict gene expression responses to perturbations')
                    ],
                    "constraints": [
                        "Limited training data",
                        "Need for biological interpretability",
                        "Computational efficiency requirements"
                    ],
                    "evaluation_metrics": data.get('task_requirements', {}).get('evaluation_criteria', {}).get('primary_metrics', ['MSE', 'Pearson correlation'])
                }
                print(f"Successfully converted analysis report to task analysis from: {target_path}")
                return task_analysis
            elif 'research_plan' in data:
                # Extract task analysis from the enhanced plan structure
                task_analysis = data['research_plan']
                print(f"Successfully loaded task analysis from: {target_path}")
                return task_analysis
            else:
                # If it's a direct task analysis file (old format)
                print(f"Successfully loaded task analysis from: {target_path}")
                return data
        else:
            print(f"Successfully loaded task analysis from: {target_path}")
            return data
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Task analysis file not found: {file_path} - {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file: {file_path} - {e}")
    except Exception as e:
        raise Exception(f"Error loading task analysis: {e}")

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
    """Main function - Auto mode"""
    print("=== Method Design Module ===")
    print("Multi-Expert Research Plan Generator (Auto Mode)\n")
    
    # Auto-load latest task analysis
    print("Automatically loading latest task analysis...")
    try:
        task_analysis = load_task_analysis(latest=True)
        print("✅ Successfully loaded latest task analysis")
    except Exception as e:
        print(f"❌ Failed to load latest task analysis: {e}")
        print("Falling back to sample data...")
        task_analysis = create_sample_task_analysis()
        print("✅ Using sample data as fallback")
    
    # Create RAG retriever
    print("\nInitializing RAG knowledge retriever...")
    rag_retriever = RAGRetriever()
    
    # Generate research plan with unified output directory
    print("\nGenerating research plan...")
    try:
        # Use unified results directory
        project_root = Path(__file__).parent.parent.parent  # scAgents root
        output_dir = str(project_root / "cellforge" / "data" / "results")
        
        # Ensure results directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
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
        
        # Show generated files with dynamic names
        if 'generated_files' in plan:
            files_info = plan['generated_files']
            base_filename = files_info['base_filename']
            print(f"\nGenerated files:")
            print(f"  - {output_dir}/{base_filename}.md (Research plan)")
            print(f"  - {output_dir}/{base_filename}.json (Detailed data)")
            print(f"  - {output_dir}/{base_filename}.mmd (Architecture diagram)")
            print(f"  - {output_dir}/{base_filename}_consensus.png (Consensus progress)")
        else:
            # Fallback to old format
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