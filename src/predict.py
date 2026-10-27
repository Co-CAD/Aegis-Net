"""Apply the supplied final N=6 model; no training or threshold selection."""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from aegis_net import CRISPR_CNNLSTM_Enhanced, extract_features, make_channels

ROOT = Path(__file__).resolve().parents[1]
N_CYCLES = 6
N_FEATURES = 23
CORE_FEATURES = [0, 8, 5]
DROPOUT = 0.3
DEVICE = "cpu"

def build_inputs(
    raw: np.ndarray,
    diff: np.ndarray,
    features: np.ndarray,
    indices: np.ndarray,
    scaler: dict[str, np.ndarray | float],
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_scaled = (raw[indices] - scaler["raw_mean"]) / scaler["raw_std"]
    diff_scaled = (diff[indices] - scaler["diff_mean"]) / scaler["diff_std"]
    sequence = np.stack([raw_scaled, diff_scaled], axis=1).astype(np.float32)
    feature_values = (
        (features[indices] - scaler["feature_mean"]) / scaler["feature_std"]
    ).astype(np.float32)
    return (
        torch.tensor(sequence, dtype=torch.float32, device=DEVICE),
        torch.tensor(feature_values, dtype=torch.float32, device=DEVICE),
    )


def load_prediction_table(path: Path) -> tuple[pd.Series, np.ndarray]:
    df = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
    columns = [f"RFU_{i}" for i in range(1, 7)]
    if df.columns.tolist() not in (columns, ["sample_id", *columns]):
        raise ValueError("Expected RFU_1,...,RFU_6, optionally preceded by sample_id; exclude labels.")
    if df.empty:
        raise ValueError("Input must contain at least one reaction.")
    ids = df["sample_id"] if "sample_id" in df else pd.Series(np.arange(1, len(df) + 1))
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Sample IDs must be non-missing and unique.")
    X = df[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("RFU values must be finite.")
    return ids, X

def predict(model_path: Path, input_path: Path, output_path: Path) -> None:
    with model_path.open("rb") as handle:
        bundle = pickle.load(handle)
    metadata = bundle["metadata"]
    if metadata["n_cycles"] != N_CYCLES:
        raise ValueError("The model bundle is not an N=6 model.")
    sample_ids, X = load_prediction_table(input_path)
    raw, diff = make_channels(X)
    manual_features = extract_features(X, N_CYCLES)
    fold_probabilities = []

    for fold in bundle["folds"]:
        rf_probability = fold["rf"].predict_proba(
            manual_features[:, CORE_FEATURES]
        )[:, 1]
        augmented = np.column_stack([manual_features, rf_probability]).astype(np.float32)
        indices = np.arange(len(X))
        X_tensor, F_tensor = build_inputs(
            raw, diff, augmented, indices, fold["scaler"]
        )
        seed_probabilities = []
        for state in fold["states"]:
            model = CRISPR_CNNLSTM_Enhanced(
                n_feats=N_FEATURES, p_drop=DROPOUT
            ).to(DEVICE)
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                probability = torch.softmax(
                    model(X_tensor, F_tensor), dim=1
                )[:, 1].cpu().numpy()
            seed_probabilities.append(probability)
        fold_probabilities.append(np.mean(seed_probabilities, axis=0))

    probability = np.mean(fold_probabilities, axis=0)
    prediction = (probability >= metadata["probability_cutoff"]).astype(int)
    result = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "probability_positive": probability,
            "prediction": prediction,
            "prediction_label": np.where(prediction == 1, "Positive", "Negative"),
            "probability_cutoff": metadata["probability_cutoff"],
            "model_version": metadata["model_version"],
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"Saved predictions: {output_path}")


def main() -> None:
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    DEVICE = args.device
    if DEVICE == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available.")
    torch.set_num_threads(1)
    model = ROOT / "models" / "aegis-net.pkl"
    if args.output.exists():
        raise FileExistsError("Choose a new output path; existing predictions are not overwritten.")
    predict(model, args.input, args.output)

if __name__ == "__main__":
    main()
