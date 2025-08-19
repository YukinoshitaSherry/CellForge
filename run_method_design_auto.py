#!/usr/bin/env python3
"""
Auto Method Design Runner
Automatically run Method Design module with latest task analysis
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # 明确指定.env文件路径
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
    print(f"Loaded .env from: {env_path}")
except ImportError:
    print("Warning: python-dotenv not installed, using system environment variables")

# Add cellforge to Python path
project_root = Path(__file__).parent
cellforge_path = project_root / "cellforge"
sys.path.insert(0, str(cellforge_path))

from Method_Design.main import load_task_analysis, RAGRetriever
from Method_Design import generate_research_plan

def main():
    """Run Method Design automatically with latest task analysis"""
    print("=== Auto Method Design Runner ===")
    print("Automatically using latest task analysis and unified output\n")
    
    try:
        # Auto-load latest task analysis
        print("Loading latest task analysis...")
        task_analysis = load_task_analysis(latest=True)
        print("✅ Successfully loaded latest task analysis")
        
        # Create RAG retriever
        print("Initializing RAG knowledge retriever...")
        rag_retriever = RAGRetriever()
        
        # Set unified output directory
        output_dir = str(project_root / "cellforge" / "data" / "results")
        
        # Ensure results directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate research plan
        print("Generating research plan...")
        plan = generate_research_plan(
            task_analysis=task_analysis,
            rag_retriever=rag_retriever,
            task_type=task_analysis.get("task_type", "gene_knockout"),
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
        print("1. Task analysis module has been run and generated results")
        print("2. All dependencies are installed")
        print("3. Output directory has write permissions")

if __name__ == "__main__":
    main()
