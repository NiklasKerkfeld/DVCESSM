from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Root folder containing method/example subdirectories
    output_folder = Path(cfg.system.evaluation_dir)
    records = []

    # 1. Parse all results.csv files across directory structure
    for csv_path in output_folder.glob("*/*/results.csv"):
        method = csv_path.parts[-3]
        example = csv_path.parts[-2]

        df = pd.read_csv(csv_path)

        # Convert Metric & Difference columns into a key-value dict
        metrics = dict(zip(df["Metric"], df["Difference"]))

        row = {"method": method, "example": example}
        for metric_name in ["IoU", "F1", "Precision", "Recall", "Adoption", "Time"]:
            val = metrics.get(metric_name)
            try:
                row[metric_name] = float(val)
            except (ValueError, TypeError):
                row[metric_name] = None

        records.append(row)

    df_all = pd.DataFrame(records)

    # 2. Compute Mean and Standard Deviation per Method
    summary = []
    target_metrics = ["IoU", "F1", "Precision", "Recall", "Adoption", "Time"]

    for method, group in df_all.groupby("method"):
        row = {"Method": method}
        for metric in target_metrics:
            if metric in group and group[metric].notna().any():
                row[f"{metric}_mean"] = round(group[metric].mean(), 4)
                row[f"{metric}_std"] = round(group[metric].std(), 4)
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
        summary.append(row)

    # 3. Export single summary file for LaTeX
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_folder / "summary_results.csv", index=False)
    
   
if __name__ == '__main__':
    main()
