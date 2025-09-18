"""
Responsibility:
- Entry point of the project.
- Takes a directory path containing config files as a CLI argument.
  Example: python main.py --config-dir configs/
- Runs the ExperimentRunner for each config file in the directory.
"""
import argparse
import os
from experiment_runner import ExperimentRunner

def main():
    parser = argparse.ArgumentParser(description="Run experiments from a directory")
    parser.add_argument("--config-dir", type=str, required=True, help="Path to directory containing config files")
    args = parser.parse_args()

    config_directory = args.config_dir

    if not os.path.isdir(config_directory):
        print(f"Error: Directory not found at {config_directory}")
        return

    # Loop through all files in the specified directory
    for filename in os.listdir(config_directory):
        # Check if the file is a YAML or JSON config file
        if filename.endswith(".yaml") or filename.endswith(".json"):
            config_path = os.path.join(config_directory, filename)
            print(f"Running experiment with config file: {config_path}")

            try:
                # Create an ExperimentRunner instance for each config file
                runner = ExperimentRunner(config_path=config_path)
                runner.run_experiment_list()
            except Exception as e:
                print(f"An error occurred while running {config_path}: {e}")

if __name__ == "__main__":
    main()