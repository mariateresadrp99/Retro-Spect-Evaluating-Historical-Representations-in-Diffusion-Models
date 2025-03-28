# analyze_anachronisms.py
# ----------------------------------------------------------
# This script computes anachronism frequency and severity scores
# from the JSON output of detect_anachronisms.py.
# It aggregates results across models and time periods.
# ----------------------------------------------------------

import json
import pandas as pd
import argparse
from collections import defaultdict


def normalize_term(term):
    """
    Normalize terms by lowercasing and simple replacements.
    For more advanced handling, fuzzy matching could be added.
    """
    return term.lower().strip()


def parse_json(json_path):
    """
    Parse annotated JSON and extract answers with model/time metadata.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    records = []
    for entry in data:
        image_path = entry["image_path"]
        prompt_text = entry.get("prompt_text", "")
        answers = entry.get("image_analysis", {})

        # Extract model and time period from image_path (customize this)
        parts = image_path.split("/")
        model = parts[-4] if len(parts) >= 4 else "unknown"
        period = parts[-3] if len(parts) >= 3 else "unknown"

        for key, response in answers.items():
            if "yes" in response.lower():
                normalized = normalize_term(key)
                records.append({
                    "Model": model,
                    "Period": period,
                    "Anachronism": normalized,
                    "Detected": 1
                })
            else:
                normalized = normalize_term(key)
                records.append({
                    "Model": model,
                    "Period": period,
                    "Anachronism": normalized,
                    "Detected": 0
                })

    return pd.DataFrame(records)


def compute_stats(df):
    """
    Compute frequency and severity of each anachronism per model/period.
    """
    grouped = df.groupby(["Model", "Period", "Anachronism"])
    summary = grouped["Detected"].agg(["sum", "count"]).reset_index()
    summary.rename(columns={"sum": "Yes_Count", "count": "Total"}, inplace=True)
    summary["Frequency"] = (summary["Yes_Count"] / summary["Total"]) * 100
    summary["Severity"] = summary["Yes_Count"] / summary["Yes_Count"].max()
    return summary


def main(json_path, output_csv):
    df = parse_json(json_path)
    stats = compute_stats(df)
    stats.to_csv(output_csv, index=False)
    print(f"Saved aggregated stats to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, help="Path to annotated anachronism JSON")
    parser.add_argument("--output_csv", required=True, help="Output CSV with frequency/severity scores")
    args = parser.parse_args()
    main(args.json_path, args.output_csv)