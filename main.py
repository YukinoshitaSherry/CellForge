
"""
cellforge Main Entry Point
End-to-End Intelligent Multi-Agent System for Automated Single-Cell Data Analysis and Method Design
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


env_file = Path(__file__).parent / ".env"
try:
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded environment variables from {env_file}")
    else:
        print(f"⚠️  .env file not found at {env_file}")
        print("   Run 'python setup_env.py' to create it")
except Exception as e:
    print(f"⚠️  Warning: Could not load .env file: {e}")
    print("   This is normal if .env file doesn't exist yet")


project_root = Path(__file__).parent
sys.path.append(str(project_root))


try:
    from cellforge.llm import LLMInterface
except ImportError:
    print("⚠️  cellforge package not found. Please run 'python install.py' first.")
    sys.exit(1)


DEFAULT_TASK_DESCRIPTION = """Your task is to develop a predictive model that accurately estimates gene expression profiles of individual K562 cells following CRISPR interference (CRISPRi), using the dataset from Norman et al. (2019, Science).

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

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration file"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        
        default_config = {
            "task_description": DEFAULT_TASK_DESCRIPTION,
            "dataset_path": "cellforge/data/datasets/",
            "output_dir": "results/",
            "llm_config": {
                "provider": "openai",  
                "model": os.getenv("MODEL_NAME", "gpt-4"),
                "api_key": "loaded_from_env"  
            },
            "workflow_phases": ["task_analysis", "method_design", "code_generation"],
            "qdrant_config": {
                "host": os.getenv("QDRANT_URL", "localhost"),
                "port": int(os.getenv("QDRANT_PORT", "6333"))
            }
        }
        
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Default configuration file created: {config_path}")
        print("⚠️  Please configure your API keys in .env file")
        print("💡 To customize your task, edit the DEFAULT_TASK_DESCRIPTION variable in main.py")
        return default_config
    
    
    config["task_description"] = DEFAULT_TASK_DESCRIPTION
    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration file completeness"""
    required_fields = ["task_description", "dataset_path", "llm_config"]
    
    for field in required_fields:
        if field not in config:
            print(f"❌ Configuration file missing required field: {field}")
            return False
    
    
    llm_api_keys = [
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("DEEPSEEK_API_KEY"),
        os.getenv("LLAMA_API_KEY"),
        os.getenv("QWEN_API_KEY")
    ]
    
    configured_llm_keys = [key for key in llm_api_keys if key and key != "your_openai_api_key_here"]
    
    if not configured_llm_keys:
        print("⚠️  No LLM API keys found in .env file")
        print("💡 Please copy env.example to .env and configure at least one LLM API key")
        return False
    
    print(f"✅ {len(configured_llm_keys)} LLM API key(s) configured")
    
    
    search_api_keys = {
        "GitHub": os.getenv("GITHUB_TOKEN"),
        "SerpAPI": os.getenv("SERPAPI_KEY"),
        "PubMed": os.getenv("PUBMED_API_KEY")
    }
    
    missing_search_keys = []
    configured_search_keys = []
    for name, key in search_api_keys.items():
        if not key or key == f"your_{name.lower()}_key_here":
            missing_search_keys.append(name)
        else:
            configured_search_keys.append(name)
    
    if missing_search_keys:
        print(f"⚠️  Missing optional search API keys: {', '.join(missing_search_keys)}")
        print("💡 These keys are optional for enhanced RAG functionality")
        if configured_search_keys:
            print(f"✅ Configured search APIs: {', '.join(configured_search_keys)}")
        else:
            print("⚠️  No search APIs configured - RAG functionality will be limited")
    
    return True

def run_task_analysis(config: Dict[str, Any]) -> bool:
    """Run Task Analysis phase"""
    try:
        print("\n" + "="*60)
        print("PHASE 1: TASK ANALYSIS")
        print("="*60)
        
        from cellforge.Task_Analysis.main import run_task_analysis
        
        
        dataset_info = {
            "dataset_path": config["dataset_path"],
            "dataset_name": "norman_2019_k562",
            "data_type": "scRNA-seq",
            "cell_line": "K562",
            "perturbation_type": "CRISPRi"
        }
        
        
        result = run_task_analysis(config["task_description"], dataset_info)
        
        if result:
            print("✅ Task analysis completed")
            return True
        else:
            print("❌ Task analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in task analysis: {str(e)}")
        return False

def run_method_design(config: Dict[str, Any]) -> bool:
    """Run Method Design phase"""
    try:
        print("\n" + "="*60)
        print("PHASE 2: METHOD DESIGN")
        print("="*60)
        
        
        from cellforge.Method_Design import generate_research_plan
        
        
        task_analysis_dir = Path("cellforge/Task_Analysis/results")
        if not task_analysis_dir.exists():
            print("❌ Task analysis results not found. Please run task analysis first.")
            return False
        
        
        task_reports = list(task_analysis_dir.glob("task_analysis_*.json"))
        if not task_reports:
            print("❌ No task analysis reports found. Please run task analysis first.")
            return False
        
        latest_report = max(task_reports, key=lambda x: x.stat().st_mtime)
        
        
        with open(latest_report, 'r', encoding='utf-8') as f:
            task_analysis = json.load(f)
        
        
        output_dir = "cellforge/data/results"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("🔧 Generating research plan...")
        plan = generate_research_plan(
            task_analysis=task_analysis,
            rag_retriever=None,  
            task_type=task_analysis.get("task_type", "gene_knockout"),
            output_dir=output_dir,
            auto_generate_code=True  
        )
        
        if plan:
            print("✅ Method design completed")
            
            
            if 'generated_files' in plan:
                files_info = plan['generated_files']
                base_filename = files_info['base_filename']
                print(f"📁 Generated files:")
                print(f"  - {output_dir}/{base_filename}.md (Research plan)")
                print(f"  - {output_dir}/{base_filename}.json (Detailed data)")
                print(f"  - {output_dir}/{base_filename}.mmd (Architecture diagram)")
                print(f"  - {output_dir}/{base_filename}_consensus.png (Consensus progress)")
                
                
                if 'code_generation' in plan:
                    code_info = plan['code_generation']
                    if code_info['status'] == 'success':
                        print(f"  - {output_dir}/result.py (Generated code)")
                        print(f"✅ Code generation completed successfully")
                    elif code_info['status'] == 'failed':
                        print(f"❌ Code generation failed: {code_info.get('error', 'Unknown error')}")
                    elif code_info['status'] == 'error':
                        print(f"❌ Code generation error: {code_info.get('error', 'Unknown error')}")
            
            return True
        else:
            print("❌ Method design failed")
            return False
        
    except Exception as e:
        print(f"❌ Error in method design: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_code_generation(config: Dict[str, Any]) -> bool:
    """Run Code Generation phase"""
    try:
        print("\n" + "="*60)
        print("PHASE 3: CODE GENERATION")
        print("="*60)
        
        
        results_dir = Path("cellforge/data/results")
        if not results_dir.exists():
            print("❌ Results directory not found. Please run method design first.")
            return False
        
        
        code_file = results_dir / "result.py"
        if code_file.exists():
            print("✅ Code generation already completed in method design phase")
            print(f"📁 Generated code: {code_file}")
            return True
        
        
        plan_files = list(results_dir.glob("research_plan_*.json"))
        if not plan_files:
            print("❌ No research plans found. Please run method design first.")
            return False
        
        latest_plan = max(plan_files, key=lambda x: x.stat().st_mtime)
        print(f"📋 Found research plan: {latest_plan}")
        
        
        try:
            from cellforge.Code_Generation import generate_code_from_plan
        except ImportError as e:
            print(f"❌ Code generation module not available: {e}")
            print("💡 Code generation requires OpenHands setup")
            return False
        
        
        print("🔧 Generating code from research plan...")
        code_file_path = generate_code_from_plan(
            research_plan=json.load(open(latest_plan, 'r', encoding='utf-8')),
            output_dir=str(results_dir)
        )
        
        if code_file_path and Path(code_file_path).exists():
            print("✅ Code generation completed")
            print(f"📁 Generated code: {code_file_path}")
            return True
        else:
            print("❌ Code generation failed")
            return False
        
    except Exception as e:
        print(f"❌ Error in code generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_complete_workflow(config: Dict[str, Any]) -> bool:
    """Run complete end-to-end workflow"""
    print("🚀 Starting cellforge End-to-End Workflow")
    print("="*80)
    
    
    if not validate_config(config):
        print("❌ Configuration validation failed, please check .env file")
        return False
    
    success = True
    
    
    for phase in config["workflow_phases"]:
        if phase == "task_analysis":
            success &= run_task_analysis(config)
        elif phase == "method_design":
            success &= run_method_design(config)
        elif phase == "code_generation":
            success &= run_code_generation(config)
    
    if success:
        print("\n" + "="*80)
        print("🎉 All phases completed!")
        print("="*80)
        print(f"Results saved to: {config['output_dir']}")
    else:
        print("\n" + "="*80)
        print("❌ Workflow execution failed")
        print("="*80)
    
    return success

def create_sample_dataset():
    """Create sample dataset directory structure"""
    print("📁 Creating sample dataset directory structure...")
    
    directories = [
        "cellforge/data/datasets/scRNA-seq",
        "cellforge/data/datasets/scATAC-seq",
        "cellforge/data/datasets/perturbation",
        "results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    
    readme_content = """

Please place your single-cell datasets in the appropriate directories:

- `scRNA-seq/`: Single-cell RNA-seq data (.h5ad files)
- `scATAC-seq/`: Single-cell ATAC-seq data (.h5ad files)  
- `perturbation/`: Drug perturbation data (.h5ad files)



Recommended AnnData format (.h5ad):
- Gene expression matrix stored in `adata.X`
- Cell metadata stored in `adata.obs`
- Gene metadata stored in `adata.var`
- Required annotations: cell type, condition, batch (if applicable)



You can download datasets from [scPerturb](https://projects.sanderlab.org/scperturb/):
- Norman et al. (2019) K562 CRISPRi data
- Adamson et al. (2016) Drug perturbation data
"""
    
    with open("cellforge/data/datasets/README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("✅ Sample dataset directory structure created")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="cellforge - Intelligent Single-Cell Analysis System")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--init", action="store_true", help="Initialize project structure")
    parser.add_argument("--phase", choices=["task_analysis", "method_design", "code_generation"], 
                       help="Run specific phase")
    
    args = parser.parse_args()
    
    if args.init:
        print("🚀 Initializing cellforge project...")
        create_sample_dataset()
        load_config(args.config)  
        print("\n✅ Project initialization completed!")
        print("📝 Please copy env.example to .env and configure your API keys")
        print("💡 To customize your task, edit the DEFAULT_TASK_DESCRIPTION variable in main.py")
        return
    
    
    config = load_config(args.config)
    
    if args.phase:
        
        if args.phase == "task_analysis":
            run_task_analysis(config)
        elif args.phase == "method_design":
            run_method_design(config)
        elif args.phase == "code_generation":
            run_code_generation(config)
    else:
        
        run_complete_workflow(config)

if __name__ == "__main__":
    main() 