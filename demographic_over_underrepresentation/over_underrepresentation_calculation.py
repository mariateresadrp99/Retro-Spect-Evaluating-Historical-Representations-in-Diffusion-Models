#!/usr/bin/env python3
"""

After you’ve obtained:
  1) LLM demographic estimates CSV (one row per prompt):
       prompt, male, female, white, black, indian,
       middle_eastern, southeast_asian, east_asian, latino
  2) FairFace‐aggregated CSV (one row per prompt_index):
       prompt_index, model_gender_male, model_gender_female,
       model_race_white, model_race_black, model_race_indian,
       model_race_middle_eastern, model_race_southeast_asian,
       model_race_east_asian, model_race_latino

This script merges them and computes under-/over‐representation for each category:
  under_d = max(0, llm_d – model_d)
  over_d  = max(0, model_d – llm_d)
"""

import pandas as pd
import argparse

def compute_under_over(llm, model):
    diff = model - llm
    return (max(0.0, llm - model), max(0.0, model - llm))

def main(llm_csv, fairface_csv, output_csv):
    # Load LLM estimates
    df_llm = pd.read_csv(llm_csv)
    # assign prompt_index by row order
    df_llm = df_llm.reset_index().rename(columns={'index':'prompt_index'})
    # rename columns to llm_*
    rename_map = {c: f'llm_{c}' for c in df_llm.columns
                  if c not in ['prompt_index','prompt']}
    df_llm = df_llm.rename(columns=rename_map)

    # Load FairFace‐aggregated demographics
    df_model = pd.read_csv(fairface_csv)

    # Merge on prompt_index
    df = df_model.merge(df_llm, on='prompt_index', how='left')

    # Define categories and their model/llm column names
    cats = {
        'male':              ('model_gender_male',   'llm_male'),
        'female':            ('model_gender_female', 'llm_female'),
        'white':             ('model_race_white',    'llm_white'),
        'black':             ('model_race_black',    'llm_black'),
        'indian':            ('model_race_indian',   'llm_indian'),
        'middle_eastern':    ('model_race_middle_eastern', 'llm_middle_eastern'),
        'southeast_asian':   ('model_race_southeast_asian','llm_southeast_asian'),
        'east_asian':        ('model_race_east_asian','llm_east_asian'),
        'latino':            ('model_race_latino',   'llm_latino'),
    }

    # Compute under_/over_ for each category
    for cat, (mcol, lcol) in cats.items():
        under_col = f'under_{cat}'
        over_col  = f'over_{cat}'
        df[[under_col, over_col]] = df.apply(
            lambda r: compute_under_over(r[lcol], r[mcol]),
            axis=1, result_type='expand'
        )

    # Select desired columns
    keep = ['prompt_index','prompt'] \
         + [m for m,_ in cats.values()] \
         + [l for _,l in cats.values()] \
         + [f'under_{c}' for c in cats] \
         + [f'over_{c}'  for c in cats]

    df[keep].to_csv(output_csv, index=False)
    print(f"Saved under/over representation metrics to {output_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--llm_csv",     required=True, help="LLM estimates CSV")
    p.add_argument("--fairface_csv",required=True, help="FairFace aggregated CSV")
    p.add_argument("--output_csv",  required=True, help="Output under/over CSV")
    args = p.parse_args()
    main(args.llm_csv, args.fairface_csv, args.output_csv)
