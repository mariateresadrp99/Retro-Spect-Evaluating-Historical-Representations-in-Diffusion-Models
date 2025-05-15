#!/usr/bin/env python3
"""
vsd_scorer.py

Visual Style Dominance (VSD) Score Calculator for HistVis.

This script computes VSD scores to quantify how much each diffusion model “defaults” to a single visual style within each historical period.

Formula:
    VSD(m, t) = max_s P_m(s | t)
  where:
    m = model
    t = historical period
    s = visual style
    P_m(s | t) = proportion of images from model m and period t labeled style s

Features:
 1. Reads a CSV of per-image style predictions:
      must contain columns: 'model', 'historical_period', 'detailed_style'
 2. Computes, for each (model, period):
      - vsd_score: the highest style proportion
      - dominant_style: which style that is
      - style_diversity: count of styles ≥ 5% of that subset
 3. Outputs a CSV of these metrics
 4. (Optional) Generates:
      - Heatmap of VSD scores
      - Bar charts by model and by period
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------------------------
# SECTION 1: CORE FUNCTIONS
# ------------------------------------------------------------------------------

def calculate_vsd_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Visual Style Dominance (VSD) scores.

    Args:
        df: Input DataFrame with columns:
            - 'model': model identifier (e.g., 'Flux_Schnell')
            - 'historical_period': period label (e.g., '1910s')
            - 'detailed_style': style label (e.g., 'photography (monochrome)')

    Returns:
        DataFrame with one row per (model, historical_period) including:
            - model
            - historical_period
            - sample_size
            - vsd_score
            - dominant_style
            - style_diversity
            - style_distribution (dict of style→proportion)
    """
    # Validate required columns
    required = {'model', 'historical_period', 'detailed_style'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    results = []
    models = df['model'].unique()
    periods = df['historical_period'].unique()

    for model in models:
        for period in periods:
            subset = df[(df['model'] == model) & (df['historical_period'] == period)]
            n = len(subset)
            if n == 0:
                continue

            # Compute proportions of each style
            counts = subset['detailed_style'].value_counts()
            props = counts / n

            # VSD = max proportion
            vsd_score = props.max()
            dominant_style = props.idxmax()

            # Count styles with ≥5% share
            style_diversity = int((props >= 0.05).sum())

            # Record full distribution (for debugging/analysis)
            style_distribution = props.to_dict()

            results.append({
                'model': model,
                'historical_period': period,
                'sample_size': n,
                'vsd_score': vsd_score,
                'dominant_style': dominant_style,
                'style_diversity': style_diversity,
                'style_distribution': style_distribution
            })

    return pd.DataFrame(results)


def analyze_vsd_results(vsd_df: pd.DataFrame) -> dict:
    """
    Derive summary statistics and rankings from VSD DataFrame.

    Args:
        vsd_df: DataFrame returned by calculate_vsd_scores()

    Returns:
        Dictionary with keys:
            - overall_vsd_stats: {'mean','median','min','max'}
            - model_vsd_rankings: avg VSD per model (desc)
            - period_vsd_rankings: avg VSD per period (desc)
            - most_biased_combination: row with highest vsd_score
            - least_biased_combination: row with lowest vsd_score
    """
    stats = {
        'mean': vsd_df['vsd_score'].mean(),
        'median': vsd_df['vsd_score'].median(),
        'min': vsd_df['vsd_score'].min(),
        'max': vsd_df['vsd_score'].max()
    }

    model_rank = vsd_df.groupby('model')['vsd_score'].mean().sort_values(ascending=False).to_dict()
    period_rank = vsd_df.groupby('historical_period')['vsd_score'].mean().sort_values(ascending=False).to_dict()

    most = vsd_df.loc[vsd_df['vsd_score'].idxmax(), ['model','historical_period','vsd_score','dominant_style']].to_dict()
    least = vsd_df.loc[vsd_df['vsd_score'].idxmin(), ['model','historical_period','vsd_score','dominant_style']].to_dict()

    return {
        'overall_vsd_stats': stats,
        'model_vsd_rankings': model_rank,
        'period_vsd_rankings': period_rank,
        'most_biased_combination': most,
        'least_biased_combination': least
    }


def generate_visualizations(vsd_df: pd.DataFrame, prefix: str):
    """
    Create and save three plots:
      1. Heatmap of VSD scores (models × periods)
      2. Bar chart of average VSD per model
      3. Bar chart of average VSD per period

    Args:
        vsd_df: VSD results DataFrame
        prefix: Filename prefix for saved images
    """
    sns.set(style="whitegrid")

    # Pivot for heatmap
    pivot = vsd_df.pivot(index='model', columns='historical_period', values='vsd_score')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1,
                cbar_kws={'label': 'VSD Score'})
    plt.title("VSD Score Heatmap")
    plt.tight_layout()
    plt.savefig(f"{prefix}_heatmap.png")
    plt.close()

    # Average VSD by model
    avg_model = vsd_df.groupby('model')['vsd_score'].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=avg_model.index, y=avg_model.values)
    plt.title("Average VSD by Model")
    plt.ylabel("Avg VSD Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{prefix}_avg_model.png")
    plt.close()

    # Average VSD by period
    avg_period = vsd_df.groupby('historical_period')['vsd_score'].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=avg_period.index, y=avg_period.values)
    plt.title("Average VSD by Historical Period")
    plt.ylabel("Avg VSD Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{prefix}_avg_period.png")
    plt.close()


# ------------------------------------------------------------------------------
# SECTION 2: COMMAND-LINE INTERFACE
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute and optionally visualize Visual Style Dominance (VSD) scores."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to CSV file with per-image style predictions."
    )
    parser.add_argument(
        "--output",
        default="vsd_results.csv",
        help="Path where the VSD results CSV will be saved."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="If set, generate and save plots of the results."
    )
    parser.add_argument(
        "--output-prefix",
        default="vsd",
        help="Prefix for plot filenames (if --visualize is set)."
    )
    args = parser.parse_args()

    # Load predictions
    print(f"Loading predictions from: {args.predictions}")
    df = pd.read_csv(args.predictions)

    # Compute VSD scores
    print("Calculating VSD scores...")
    vsd_df = calculate_vsd_scores(df)

    # Save the results
    print(f"Saving VSD results to: {args.output}")
    # Convert dict column to string for CSV compatibility
    vsd_df['style_distribution'] = vsd_df['style_distribution'].apply(str)
    vsd_df.to_csv(args.output, index=False)

    # Summary analysis
    print("▶ Generating summary analysis...")
    analysis = analyze_vsd_results(vsd_df)

    # Print key findings
    print("\n=== VSD SUMMARY ===")
    print(f"Overall VSD: mean={analysis['overall_vsd_stats']['mean']:.3f}, "
          f"range=[{analysis['overall_vsd_stats']['min']:.3f}, "
          f"{analysis['overall_vsd_stats']['max']:.3f}]")
    print("Top models by avg VSD:")
    for model, score in analysis['model_vsd_rankings'].items():
        print(f"  {model}: {score:.3f}")
    print("Top periods by avg VSD:")
    for period, score in analysis['period_vsd_rankings'].items():
        print(f"  {period}: {score:.3f}")

    # Optional plotting
    if args.visualize:
        print("Creating visualizations...")
        generate_visualizations(vsd_df, args.output_prefix)
        print("Plots saved.")

    print("All done.")

if __name__ == "__main__":
    main()
