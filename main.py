"""
Responsibility:
- Entry point of the project.
- Takes config.yaml/json as CLI argument in this format:
  python main.py --config configs/exp1.yaml
- Runs the ExperimentRunner.
"""
import argparse
from experiment_runner import ExperimentRunner

def main():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    runner = ExperimentRunner(config_path=args.config)
    runner.run_experiment_list()

if __name__ == "__main__":
    main()
