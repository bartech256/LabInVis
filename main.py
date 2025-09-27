"""
Responsibility:
- Entry point of the project.
- Runs experiments for each config file in a specified directory.
  Example usage:
      python main.py --config-dir configs/
"""

import argparse
import os
from experiment_runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run experiments from a config directory")
    parser.add_argument(
        "--config-dir",
        type=str,
        required=True,
        help="Path to directory containing YAML or JSON config files"
    )
    args = parser.parse_args()
    config_directory = args.config_dir

    if not os.path.isdir(config_directory):
        print(f"Error: Directory not found at '{config_directory}'")
        return

    # List all YAML/JSON config files
    config_files = [
        f for f in os.listdir(config_directory)
        if f.endswith(".yaml") or f.endswith(".yml") or f.endswith(".json")
    ]

    if not config_files:
        print(f"No config files found in directory '{config_directory}'")
        return

    for filename in sorted(config_files):
        config_path = os.path.join(config_directory, filename)
        print(f"\n--- Running experiment: {filename} ---")

        try:
            runner = ExperimentRunner(config_path=config_path)
            runner.run_experiment_list()
        except Exception as e:
            print(f"Error running experiment '{filename}': {e}")


if __name__ == "__main__":
    main()
