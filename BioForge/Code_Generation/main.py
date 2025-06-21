#!/usr/bin/env python3
"""
Code Generation Main Controller
Integrates OpenHands for automatic code generation from Method Design plans
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from agentic_coder import AgenticCoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeGenerator:
    """
    Main code generation controller
    """
    
    def __init__(self, output_base_dir: str = "BioForge/Code_Generation/Designed_models"):
        self.output_base_dir = output_base_dir
        self.coder = AgenticCoder()
        
    def find_latest_plan(self, plan_dir: str = "Method_Design/results") -> Optional[str]:
        """
        Find the latest Method Design plan file
        """
        try:
            plan_path = Path(plan_dir)
            if not plan_path.exists():
                logger.error(f"Plan directory not found: {plan_dir}")
                return None
            
            # Look for JSON files
            json_files = list(plan_path.glob("*.json"))
            if not json_files:
                logger.error(f"No JSON plan files found in {plan_dir}")
                return None
            
            # Get the most recent file
            latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Found latest plan: {latest_file}")
            return str(latest_file)
            
        except Exception as e:
            logger.error(f"Error finding latest plan: {str(e)}")
            return None
    
    def create_output_directory(self, plan_data: Dict[str, Any]) -> str:
        """
        Create output directory based on plan metadata
        """
        try:
            # Extract metadata
            metadata = plan_data.get('report_metadata', {})
            task_type = metadata.get('task_type', 'unknown_task')
            dataset = metadata.get('dataset', 'unknown_dataset')
            
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create directory name
            dir_name = f"{task_type}_{dataset}_{timestamp}".replace(" ", "_").replace("(", "").replace(")", "")
            output_dir = os.path.join(self.output_base_dir, dir_name)
            
            # Create directory
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
            
            return output_dir
            
        except Exception as e:
            logger.error(f"Error creating output directory: {str(e)}")
            return os.path.join(self.output_base_dir, f"generated_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    def validate_plan(self, plan_data: Dict[str, Any]) -> bool:
        """
        Validate the Method Design plan
        """
        try:
            required_sections = ['report_metadata', 'data_preprocessing', 'model_design', 'training_strategy']
            
            for section in required_sections:
                if section not in plan_data:
                    logger.error(f"Missing required section: {section}")
                    return False
            
            # Check metadata
            metadata = plan_data['report_metadata']
            if not metadata.get('task_type') or not metadata.get('dataset'):
                logger.error("Missing required metadata: task_type or dataset")
                return False
            
            logger.info("Plan validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating plan: {str(e)}")
            return False
    
    def generate_code(self, plan_file: Optional[str] = None) -> bool:
        """
        Main function to generate code from Method Design plan
        """
        try:
            # Find plan file if not provided
            if not plan_file:
                plan_file = self.find_latest_plan()
                if not plan_file:
                    logger.error("No plan file found")
                    return False
            
            # Load plan data
            logger.info(f"Loading plan from: {plan_file}")
            with open(plan_file, 'r') as f:
                plan_data = json.load(f)
            
            # Validate plan
            if not self.validate_plan(plan_data):
                logger.error("Plan validation failed")
                return False
            
            # Create output directory
            output_dir = self.create_output_directory(plan_data)
            
            # Generate code using the agentic coder
            logger.info("Starting code generation with Agentic Coder...")
            success = self.coder.generate_code_from_plan(plan_file, output_dir)
            
            if success:
                logger.info("✅ Code generation completed successfully!")
                logger.info(f"📁 Output directory: {output_dir}")
                
                # Create summary report
                self.create_summary_report(plan_data, output_dir)
                
                return True
            else:
                logger.error("❌ Code generation failed!")
                return False
                
        except Exception as e:
            logger.error(f"Error in code generation: {str(e)}")
            return False
    
    def create_summary_report(self, plan_data: Dict[str, Any], output_dir: str):
        """
        Create a summary report of the generated code
        """
        try:
            metadata = plan_data.get('report_metadata', {})
            
            summary = {
                "generation_info": {
                    "timestamp": datetime.now().isoformat(),
                    "plan_source": metadata.get('title', 'Unknown'),
                    "task_type": metadata.get('task_type', 'Unknown'),
                    "dataset": metadata.get('dataset', 'Unknown')
                },
                "generated_files": [
                    "main.py - Main execution script",
                    "model.py - Model architecture definition", 
                    "data_loader.py - Data preprocessing and loading",
                    "train.py - Training functions",
                    "evaluate.py - Evaluation and metrics",
                    "utils.py - Utility functions",
                    "config.py - Configuration management",
                    "requirements.txt - Dependencies"
                ],
                "next_steps": [
                    "1. Review the generated code in the output directory",
                    "2. Install dependencies: pip install -r requirements.txt",
                    "3. Configure your data paths in config.py",
                    "4. Run the analysis: python main.py",
                    "5. Monitor training progress and results"
                ],
                "plan_summary": {
                    "data_preprocessing_steps": len(plan_data.get('data_preprocessing', {}).get('steps', [])),
                    "model_components": len(plan_data.get('model_design', {}).get('key_components', [])),
                    "training_parameters": plan_data.get('training_strategy', {}).get('training_parameters', {})
                }
            }
            
            # Save summary
            summary_path = os.path.join(output_dir, "generation_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Create markdown summary
            md_summary = f"""# Code Generation Summary

## Generation Info
- **Timestamp**: {summary['generation_info']['timestamp']}
- **Plan Source**: {summary['generation_info']['plan_source']}
- **Task Type**: {summary['generation_info']['task_type']}
- **Dataset**: {summary['generation_info']['dataset']}

## Generated Files
"""
            for file in summary['generated_files']:
                md_summary += f"- {file}\n"
            
            md_summary += """
## Next Steps
"""
            for step in summary['next_steps']:
                md_summary += f"{step}\n"
            
            md_summary += f"""
## Plan Summary
- **Data Preprocessing Steps**: {summary['plan_summary']['data_preprocessing_steps']}
- **Model Components**: {summary['plan_summary']['model_components']}
- **Training Parameters**: {summary['plan_summary']['training_parameters']}

## Usage Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Data Paths**:
   Edit `config.py` to set your data file paths

3. **Run Analysis**:
   ```bash
   python main.py
   ```

4. **Monitor Progress**:
   Check the logs and output files for training progress

## Notes
- The generated code follows the exact specifications from the Method Design plan
- All components are modular and can be customized as needed
- Error handling and logging are included for robust execution
- The code is ready for immediate execution with proper data setup
"""
            
            md_path = os.path.join(output_dir, "README.md")
            with open(md_path, 'w') as f:
                f.write(md_summary)
            
            logger.info(f"Summary reports created: {summary_path}, {md_path}")
            
        except Exception as e:
            logger.error(f"Error creating summary report: {str(e)}")
    
    def list_generated_models(self) -> List[str]:
        """
        List all generated models
        """
        try:
            base_path = Path(self.output_base_dir)
            if not base_path.exists():
                return []
            
            models = [d.name for d in base_path.iterdir() if d.is_dir()]
            return sorted(models, reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing generated models: {str(e)}")
            return []
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific generated model
        """
        try:
            model_path = Path(self.output_base_dir) / model_name
            if not model_path.exists():
                logger.error(f"Model not found: {model_name}")
                return None
            
            # Load summary
            summary_path = model_path / "generation_summary.json"
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return None

def main():
    """
    Main function for command-line usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Code Generation Controller")
    parser.add_argument("--plan", help="Path to Method Design plan JSON file (optional, will use latest if not provided)")
    parser.add_argument("--output-dir", default="BioForge/Code_Generation/Designed_models", help="Base output directory")
    parser.add_argument("--list", action="store_true", help="List all generated models")
    parser.add_argument("--info", help="Get info about specific model")
    
    args = parser.parse_args()
    
    # Initialize code generator
    generator = CodeGenerator(args.output_dir)
    
    if args.list:
        models = generator.list_generated_models()
        print("Generated Models:")
        for model in models:
            print(f"  - {model}")
        return
    
    if args.info:
        info = generator.get_model_info(args.info)
        if info:
            print(json.dumps(info, indent=2))
        else:
            print(f"Model not found: {args.info}")
        return
    
    # Generate code
    success = generator.generate_code(args.plan)
    
    if success:
        print("✅ Code generation completed successfully!")
        print("\nNext steps:")
        print("1. Check the generated code in the output directory")
        print("2. Install dependencies and run the analysis")
    else:
        print("❌ Code generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 