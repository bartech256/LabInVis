import os
import json
import yaml
import csv
import glob


def get_metrics_and_configs(base_folder):
    """
    Extracts metrics and configurations from 'exp_*' and 'catboost_*' folders
    and saves them to a CSV.

    Args:
        base_folder (str): The path to the base directory containing the
                           experiment folders.
    """

    # Define the output CSV file path
    output_csv_file = os.path.join(base_folder, "experiment_summary.csv")

    data_to_write = []

    # Get a list of all folders starting with 'exp' or 'catboost'
    experiment_folders = glob.glob(os.path.join(base_folder, "exp*")) + \
                         glob.glob(os.path.join(base_folder, "catboost*"))

    # Define the header for the CSV file
    headers = [
        "Experiment_ID",
        "Validation_scaled_MSE",
        "Test_scaled_MSE",
        "GNN_model_type",
        "radius1",
        "radius1_k",
        "radius2",
        "radius2_k",
        "radius3_k"
    ]

    data_to_write.append(headers)

    for folder in experiment_folders:
        folder_name = os.path.basename(folder)

        metrics_file = os.path.join(folder, "metrics.json")
        config_file = os.path.join(folder, "config.yaml")

        validation_mse = None
        test_mse = None
        gnn_model_type = None
        radius1 = None
        radius1_k = None
        radius2 = None
        radius2_k = None
        radius3_k = None

        # 1. Extract metrics from metrics.json (common to both)
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    validation_mse = metrics_data.get("Validation", {}).get("scaled", {}).get("MSE")
                    test_mse = metrics_data.get("Test", {}).get("scaled", {}).get("MSE")
            except Exception as e:
                print(f"Error reading {metrics_file}: {e}")
                continue

        # 2. Extract configurations based on folder type
        if folder_name.startswith("exp"):
            # This is an "exp" folder, so we expect a config file
            gnn_model_type = "GNN"  # Default value for clarity
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        config_data = yaml.safe_load(f)
                        gnn_model_type = config_data.get("GNN_model_type")
                        radius1 = config_data.get("radius1")
                        radius1_k = config_data.get("radius1_k")
                        radius2 = config_data.get("radius2")
                        radius2_k = config_data.get("radius2_k")
                        radius3_k = config_data.get("radius3_k")
                except Exception as e:
                    print(f"Error reading {config_file}: {e}")
                    continue
        elif folder_name.startswith("catboost"):
            # This is a "catboost" folder; set a fixed model type
            gnn_model_type = "CatBoost"

        # Create a row with the extracted data
        row = [
            folder_name,
            validation_mse,
            test_mse,
            gnn_model_type,
            radius1,
            radius1_k,
            radius2,
            radius2_k,
            radius3_k
        ]
        data_to_write.append(row)

    # Write the data to the CSV file
    try:
        with open(output_csv_file, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(data_to_write)
        print(f"✅ Data successfully written to {output_csv_file}")
    except PermissionError:
        print(
            f"❌ Permission denied: Cannot write to '{output_csv_file}'. Please run the script from a directory with write access.")
    except Exception as e:
        print(f"An error occurred while writing the CSV: {e}")


# Example usage:
if __name__ == '__main__':
    base_folder_path = "C:\\Users\\bar25\\OneDrive\\Documents\\Technion\\GIT code\\lab in vis\\LabInVis\\experiments"
    get_metrics_and_configs(base_folder_path)