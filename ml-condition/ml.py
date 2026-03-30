import argparse
import datetime
import os

import numpy as np
import pandas as pd
from rxnopt import ReactionOptimizer

VIRTUAL_TIME = 7.25

def get_search_space(df):
    """Defines the search space for the optimization"""
    muts = df["mut"].dropna().unique().tolist()
    enes = [1.0, 2.5, 5.0]
    acids = [5.0]
    organs = [5.0, 7.5]
    temps = [5, 10, 25]
    # Ensure VIRTUAL_TIME is in the search space
    times = np.arange(0.25, 7.5, 0.25).tolist()
    if VIRTUAL_TIME not in times:
        times.append(VIRTUAL_TIME)
    
    powers = np.arange(3, 16, 3).tolist()

    # Store everything as strings because rxnopt casts prev_rxn_info to strings during comparison
    condition_dict = {
        "mut": [str(x) for x in muts],
        "ene": [str(float(x)) for x in enes],
        "acid": [str(float(x)) for x in acids],
        "organ": [str(float(x)) for x in organs],
        "temp": [str(int(x)) for x in temps],
        "time": [str(float(x)) for x in times],
        "power": [str(int(x)) for x in powers],
    }
    return condition_dict


def get_descriptors(condition_dict):
    """Generate descriptors for the search space."""
    desc_dict = {}

    # 1. mut: one-hot encoding
    muts = condition_dict["mut"]
    mut_desc = pd.get_dummies(muts, dtype=float)
    mut_desc.index = muts
    desc_dict["mut"] = mut_desc

    # 2. Others: numerical values as their own descriptors
    for k, v in condition_dict.items():
        if k == "mut":
            continue
        # Use the float representation as the descriptor vector
        desc_dict[k] = pd.DataFrame(
            [float(val) for val in v], columns=[k], index=v, dtype=float
        )

    return desc_dict


def main():
    parser = argparse.ArgumentParser(description="Reaction Optimization with rxnopt")
    parser.add_argument(
        "--surrogate_model",
        type=str,
        default="GP",
        help="Surrogate model to use (e.g., GP, RF)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature parameter for exploration",
    )
    args = parser.parse_args()

    print("-" * 60)
    print(" Reaction Optimization using Bayesian Optimization (reactionopt)")
    print("-" * 60)

    # 1. Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "exp_result.csv")

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # 2. Preprocess Data
    if df["result"].dtype == object and df["result"].str.contains("%").any():
        df["result"] = df["result"].str.replace("%", "", regex=False).astype(float)

    if "batch" not in df.columns:
        df["batch"] = 0

    # Cast df features to string to match condition_dict format exactly
    # otherwise load_prev_rxn stringification rules might subtly mismatch e.g., '5' vs '5.0'
    df["ene"] = df["ene"].astype(float).astype(str)
    df["acid"] = df["acid"].astype(float).astype(str)
    df["organ"] = df["organ"].astype(float).astype(str)
    df["temp"] = df["temp"].astype(int).astype(str)
    df["time"] = df["time"].astype(float).astype(str)
    df["power"] = df["power"].astype(int).astype(str)
    df["mut"] = df["mut"].astype(str)

    # Inject virtual data points
    print(
        f"Injecting virtual data points at time={VIRTUAL_TIME} with result=0.0 (Decomposition Constraint)..."
    )

    group_cols = ["mut", "ene", "acid", "organ", "temp", "power"]
    virtual_df = df[group_cols].drop_duplicates().copy()

    virtual_df["time"] = str(VIRTUAL_TIME)
    virtual_df["result"] = 0.0
    virtual_df["batch"] = -1

    for col in df.columns:
        if col not in virtual_df.columns:
            # If the column is string/object, use a placeholder
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                virtual_df[col] = "virtual_decay"
            else:
                virtual_df[col] = 0

    df = pd.concat([df, virtual_df], ignore_index=True)
    print(f"Added {len(virtual_df)} virtual data points.")

    # 3. Setup Optimizer
    condition_dict = get_search_space(df)
    desc_dict = get_descriptors(condition_dict)

    opt_metrics = ["result"]
    opt_settings = [{"opt_direct": "max", "opt_range": [0, 100], "metric_weight": 1.0}]

    output_dir = os.path.join(script_dir, "optimization_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reaction_optimizer = ReactionOptimizer(
        opt_metrics=opt_metrics,
        opt_metric_settings=opt_settings,
        opt_type="auto",
        save_dir=output_dir,
    )

    # 4. Load Configuration
    reaction_optimizer.load_rxn_space(condition_dict=condition_dict)
    reaction_optimizer.load_desc(desc_dict=desc_dict)

    # 5. Load Previous Data
    reaction_optimizer.load_prev_rxn(df)

    # 6. Run Optimization
    print(
        f"Running optimization (batch_size=5, surrogate_model={args.surrogate_model})..."
    )
    reaction_optimizer.optimize(
        batch_size=5,
        desc_normalize="minmax",
        surrogate_model=args.surrogate_model,
        temperature=args.temperature,  # 调整这里的 temperature 参数 (0.0=纯利用，越大约倾向探索)
    )

    # 7. Save and Display Results
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    model_name = args.surrogate_model
    suffix = f"{model_name}_{timestamp}"

    reaction_optimizer.save_results(filetype="csv", suffix=suffix)

    print("\nRecommended Conditions for Next Batch:")
    recs_df = pd.DataFrame(
        reaction_optimizer.selected_conditions,
        columns=reaction_optimizer.condition_types,
    )

    if getattr(reaction_optimizer, "pred_mean", None) is not None:
        preds = reaction_optimizer.pred_mean
        recs_df["predicted_result"] = preds if len(preds.shape) == 1 else preds[:, 0]

        if getattr(reaction_optimizer, "pred_std", None) is not None:
            uncertainties = reaction_optimizer.pred_std
            recs_df["uncertainty"] = (
                uncertainties if len(uncertainties.shape) == 1 else uncertainties[:, 0]
            )

    print(recs_df.to_string())

    print(recs_df.to_string())

if __name__ == "__main__":
    main()
