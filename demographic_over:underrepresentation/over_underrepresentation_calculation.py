# --------------------------------------------------------------
# Compares TTI model outputs from a demographic classifier
# (FairFace, etc.) with LLM-based historical demographic estimates.
# Computes over- and under-representation metrics.
# --------------------------------------------------------------
import json
import argparse
import pandas as pd

def load_llm_estimates(llm_path):
    """
    Load the LLM demographic estimates.
    Format: [{
        'index': 0,
        'prompt': 'A person cooking in the 19th century',
        'gender_estimate': { 'male': 60, 'female': 40 },
        'race_estimate': { 'white': 50, 'black': 20, 'asian': 20, 'indian': 10 }
    }, ...]
    """
    with open(llm_path, "r") as f:
        return json.load(f)

def load_model_outputs(model_csv):
    """
    Load a CSV that contains demographic distribution per prompt.
    For example:
        prompt_index, model_gender_male, model_gender_female, model_race_white, model_race_black, ...
    Values might be raw counts or percentages. For this example, we'll assume percentages.
    """
    df = pd.read_csv(model_csv)
    return df

def compute_over_under(llm_val, model_val):
    """
    Return (under, over) for a single category, following Equations (1) and (2).
    llm_val and model_val are numeric percentages.
    """
    if model_val <= llm_val:
        under = llm_val - model_val
        over = 0
    else:
        under = 0
        over = model_val - llm_val
    return under, over

def main(llm_path, model_csv, output_csv):
    llm_data = load_llm_estimates(llm_path)
    df_model = load_model_outputs(model_csv)

    # Convert llm_data into a DataFrame keyed by 'index'
    # We'll keep columns for each category: male, female, white, black, asian, indian
    records = []
    for entry in llm_data:
        idx = entry['index']
        prompt = entry['prompt']
        gender_est = entry.get('gender_estimate', {})
        race_est = entry.get('race_estimate', {})
        records.append({
            'index': idx,
            'prompt': prompt,
            'llm_gender_male': gender_est.get('male', 0),
            'llm_gender_female': gender_est.get('female', 0),
            'llm_race_white': race_est.get('white', 0),
            'llm_race_black': race_est.get('black', 0),
            'llm_race_asian': race_est.get('asian', 0),
            'llm_race_indian': race_est.get('indian', 0)
        })
    df_llm = pd.DataFrame(records)

    # Merge model outputs with LLM estimates by 'index'
    df_merged = pd.merge(df_model, df_llm, on='index', how='left')

    # We'll compute under/over for each of the 6 categories
    categories = [
        ('gender_male', 'llm_gender_male'),
        ('gender_female', 'llm_gender_female'),
        ('race_white', 'llm_race_white'),
        ('race_black', 'llm_race_black'),
        ('race_asian', 'llm_race_asian'),
        ('race_indian', 'llm_race_indian')
    ]

    under_cols = []
    over_cols = []

    for model_col, llm_col in categories:
        under_col = f'under_{model_col}'
        over_col = f'over_{model_col}'
        under_cols.append(under_col)
        over_cols.append(over_col)

        # We assume the model CSV has columns named exactly like 'model_gender_male'
        df_merged[under_col], df_merged[over_col] = zip(*df_merged.apply(
            lambda row: compute_over_under(
                row[llm_col],
                row[f"model_{model_col}"]
            ), axis=1
        ))

    df_merged.to_csv(output_csv, index=False)
    print(f" Over/Under representation analysis saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_path", required=True, help="Path to LLM-based demographic estimates (JSON)")
    parser.add_argument("--model_csv", required=True, help="CSV with model's demographic distribution by prompt_index")
    parser.add_argument("--output_csv", required=True, help="CSV to save final comparison with under/over representation")
    args = parser.parse_args()
    main(args.llm_path, args.model_csv, args.output_csv)
