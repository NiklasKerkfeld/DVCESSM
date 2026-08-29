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

    if not records:
        print(f"No results.csv files found under {output_folder}")
        return

    df_all = pd.DataFrame(records)

    # 2. Compute Mean, Std, and pre-formatted LaTeX string per Method
    summary = []
    target_metrics = ["IoU", "F1", "Precision", "Recall", "Adoption", "Time"]

    for method, group in df_all.groupby("method"):
        row = {"method": method}
        for metric in target_metrics:
            if metric in group and group[metric].notna().any():
                m = group[metric].mean()
                s = group[metric].std()

                # Raw floats for analysis/plotting
                row[f"{metric}_mean"] = round(m, 4)
                row[f"{metric}_std"] = round(s, 4) if pd.notna(s) else 0.0

                # Formatted LaTeX string column for direct table rendering
                std_val = s if pd.notna(s) else 0.0
                row[metric] = f"${m:.3f} (\\pm {std_val:.3f})$"
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                row[metric] = "-"
        summary.append(row)

    # 3. Export summary CSV for LaTeX
    summary_df = pd.DataFrame(summary)
    summary_path = output_folder / "summary_results.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary generated: {summary_path}")


if __name__ == "__main__":
    main()
