"""
QAOA-Based Feature Selection for the Desertification CNN Classifier
=====================================================================

Purpose
-------
Your CNN (Model.h5) classifies satellite tiles into Cloudy / Desert /
Green Area / Water but is currently performing close to random
(ROC AUC ~0.50). This script adds a quantum-classical feature-selection
step: it pulls the CNN's learned embedding for each image, shortlists
the most informative dimensions classically, then uses QAOA to pick
the best non-redundant subset of those dimensions before retraining a
lightweight classifier on top.

Why not raw pixels or QIP? Same reasoning as your paper's QIP section:
qubit counts. QAOA here only needs one qubit per *candidate feature*,
not per pixel, so a shortlist of ~20 embedding dimensions is entirely
feasible on a simulator (and even near-term hardware).

Install once (Colab):
    !pip install qiskit qiskit-optimization qiskit-algorithms tensorflow scikit-learn pillow numpy

Pipeline
--------
1. Load Model.h5, cut it at the penultimate layer -> feature extractor.
2. Run your existing dataset through it to get an embedding per image.
3. Classical pre-filter (mutual information) -> shortlist of ~20 dims.
   (Keeps the QUBO/QAOA problem at a realistic qubit count.)
4. Build a QUBO: reward relevance to the label, penalize redundancy
   between features, constrain the subset to K features.
5. Solve the QUBO with QAOA (qiskit_optimization + qiskit_algorithms).
6. Train + evaluate a classifier on the QAOA-selected features.
7. Train the same classifier on the full embedding as a baseline, so
   you can report a direct before/after comparison (mirrors your
   Section 4.2 ROC/AUC evaluation).
"""

import argparse
import os
import time
import numpy as np
from PIL import Image

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler


# =====================================================================
# CONFIG — the knobs you'll actually want to tune
# =====================================================================
MODEL_PATH = "Model.h5"
DATA_ROOT = os.environ.get("DATA_ROOT", "/content/data")

# Which layer of your CNN to treat as the "embedding". -2 = the layer
# right before the final classification/softmax layer. Run
# full_model.summary() once and adjust if your architecture differs.
EMBEDDING_LAYER_INDEX = -2

# Your existing data folder structure (from the original notebook).
LABELS = {
    "cloudy": "Cloudy",
    "desert": "Desert",
    "green_area": "Green Area",
    "water": "Water",
}

SHORTLIST_SIZE = 20   # candidate features handed to QAOA (keep <= ~25)
K = 8                 # how many features QAOA should ultimately select
ALPHA = 1.0           # weight on relevance (reward for informative features)
BETA = 0.5            # weight on redundancy (penalty for correlated features)
PENALTY = 2.0         # weight enforcing the cardinality constraint sum(x)=K
QAOA_REPS = 2         # QAOA circuit depth (p)
QAOA_MAXITER = 200    # classical optimizer iterations


# =====================================================================
# STEP 1: Load the trained CNN and turn it into a feature extractor
# =====================================================================
def build_feature_extractor(model_path=MODEL_PATH, layer_index=EMBEDDING_LAYER_INDEX):
    try:
        from tensorflow.keras.models import load_model, Model as KerasModel
    except ImportError as error:
        raise ImportError(
            "TensorFlow is required to load Model.h5. Install a TensorFlow build "
            "compatible with the active Python interpreter."
        ) from error

    full_model = load_model(model_path)
    full_model.summary()  # inspect once to confirm the embedding layer

    extractor = KerasModel(
        inputs=full_model.inputs[0],
        outputs=full_model.layers[layer_index].output,
    )
    print("Embedding output shape:", extractor.output_shape)
    return full_model, extractor


# =====================================================================
# STEP 2: Load images and extract embeddings + labels
# =====================================================================
def load_dataset_embeddings(extractor, input_size, data_root=DATA_ROOT, labels=LABELS,
                            batch_size=32):
    started = time.perf_counter()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STEP 2 START: extracting CNN embeddings")
    image_paths, y_labels = [], []
    for relative_folder, label in labels.items():
        folder = os.path.join(data_root, relative_folder)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}. "
                "Pass the dataset parent with --data-root or DATA_ROOT."
            )
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            try:
                with Image.open(fpath) as img:
                    img.convert("RGB").resize(input_size)
                image_paths.append(fpath)
                y_labels.append(label)
            except Exception as e:
                print(f"Skipping {fpath}: {e}")

    y_labels = np.array(y_labels)

    if len(image_paths) == 0:
        raise ValueError(f"No readable images found below {data_root}.")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)

    def image_batches():
        for start in range(0, len(image_paths), batch_size):
            batch = []
            for fpath in image_paths[start:start + batch_size]:
                with Image.open(fpath) as img:
                    batch.append(
                        np.asarray(img.convert("RGB").resize(input_size), dtype=np.float32)
                        / 255.0
                    )
            # Keras 3 requires generator inputs to be wrapped in a tuple.
            yield (np.asarray(batch),)

    embeddings = extractor.predict(
        image_batches(), steps=int(np.ceil(len(image_paths) / batch_size)), verbose=1
    )
    if embeddings.ndim != 2:
        raise ValueError(
            f"Embedding layer must produce a 2-D array, got shape {embeddings.shape}. "
            "Choose a flattened or dense embedding layer."
        )
    print("Embeddings shape:", embeddings.shape)  # (num_images, embedding_dim)
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STEP 2 END: "
        f"embedding extraction took {time.perf_counter() - started:.1f}s"
    )

    return embeddings, y_encoded, le


# =====================================================================
# STEP 3: Classical pre-filter to a QAOA-feasible shortlist
# =====================================================================
def shortlist_features(embeddings, y_encoded, shortlist_size=SHORTLIST_SIZE):
    mi_scores = mutual_info_classif(embeddings, y_encoded, random_state=42)
    shortlist_idx = np.argsort(mi_scores)[::-1][:shortlist_size]

    X_shortlist = embeddings[:, shortlist_idx]
    relevance = mi_scores[shortlist_idx]
    relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-9)

    redundancy = np.abs(np.corrcoef(X_shortlist.T))
    np.fill_diagonal(redundancy, 0.0)

    return shortlist_idx, relevance, redundancy


# =====================================================================
# STEP 4 + 5: Build the QUBO and solve it with QAOA
# =====================================================================
def qaoa_select_features(relevance, redundancy, k=K, alpha=ALPHA, beta=BETA,
                          penalty=PENALTY, reps=QAOA_REPS, maxiter=QAOA_MAXITER,
                          seed=42):
    n = len(relevance)
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and the shortlist size ({n}), got {k}.")

    qp = QuadraticProgram(name="feature_selection")
    for i in range(n):
        qp.binary_var(name=f"x{i}")

    # Minimize:  -alpha * sum(relevance_i * x_i)
    #          + beta  * sum_{i<j} redundancy_ij * x_i * x_j
    #          + penalty * (sum(x_i) - k)^2
    linear = {f"x{i}": -alpha * relevance[i] + penalty * (1 - 2 * k) for i in range(n)}
    quadratic = {
        (f"x{i}", f"x{j}"): beta * redundancy[i, j] + 2 * penalty
        for i in range(n) for j in range(i + 1, n)
    }
    qp.minimize(linear=linear, quadratic=quadratic)

    sampler = StatevectorSampler(seed=seed)
    optimizer = COBYLA(maxiter=maxiter)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)
    meo = MinimumEigenOptimizer(qaoa)

    started = time.perf_counter()
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STEP 5 START: "
        f"QAOA ({n} qubits, reps={reps}, maxiter={maxiter})"
    )
    result = meo.solve(qp)
    selected_mask = np.array([bool(round(v)) for v in result.x])
    if selected_mask.sum() != k:
        raise RuntimeError(
            f"QAOA returned {selected_mask.sum()} features, expected exactly {k}. "
            "Tune PENALTY or increase QAOA_MAXITER before comparing methods."
        )

    print(
        f"QAOA selected {selected_mask.sum()} / {n} shortlisted features; "
        f"objective={result.fval:.6f}; "
        f"STEP 5 END after {time.perf_counter() - started:.1f}s"
    )
    return selected_mask


# =====================================================================
# STEP 6 + 7: Cross-validated comparison across four settings:
#   A) QAOA-selected features           <- what you actually want to test
#   B) Classical top-K (no redundancy term, no QAOA) <- isolates QAOA's value
#   C) Random K features (averaged over several draws) <- sanity floor
#   D) Full embedding, no selection at all             <- your current baseline
# =====================================================================
def _fold_auc(X_train, X_test, y_train, y_test, random_state):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    clf.fit(X_train_scaled, y_train)
    probabilities = clf.predict_proba(X_test_scaled)
    if len(np.unique(y_train)) > 2:
        return roc_auc_score(y_test, probabilities, multi_class="ovr")
    return roc_auc_score(y_test, probabilities[:, 1])


def _training_shortlist(X_train, y_train, shortlist_size):
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    shortlist_idx = np.argsort(mi_scores)[::-1][:shortlist_size]
    X_shortlist = X_train[:, shortlist_idx]
    relevance = mi_scores[shortlist_idx]
    relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-9)
    redundancy = np.abs(np.corrcoef(X_shortlist.T))
    redundancy = np.nan_to_num(redundancy, nan=0.0)
    np.fill_diagonal(redundancy, 0.0)
    return shortlist_idx, relevance, redundancy


def _cross_validated_selector_auc(embeddings, y_encoded, selector, k=K,
                                  n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(embeddings, y_encoded)):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        selected_idx = selector(X_train, y_train, fold)
        scores.append(_fold_auc(
            X_train[:, selected_idx], X_test[:, selected_idx],
            y_train, y_test, random_state + fold,
        ))
    return np.mean(scores), np.std(scores), scores


def run_full_comparison(embeddings, y_encoded, k=K, shortlist_size=SHORTLIST_SIZE,
                        n_random_draws=10, qaoa_maxiter=QAOA_MAXITER,
                        qaoa_reps=QAOA_REPS,
                        random_state=42):
    def qaoa_selector(X_train, y_train, fold):
        shortlist_idx, relevance, redundancy = _training_shortlist(
            X_train, y_train, shortlist_size
        )
        selected_mask = qaoa_select_features(
            relevance, redundancy, k=k, maxiter=qaoa_maxiter,
            reps=qaoa_reps,
            seed=random_state + fold
        )
        return shortlist_idx[selected_mask]

    def classical_selector(X_train, y_train, _fold):
        shortlist_idx, _, _ = _training_shortlist(X_train, y_train, shortlist_size)
        return shortlist_idx[:k]

    rng = np.random.default_rng(random_state)

    def random_selector(X_train, y_train, _fold):
        shortlist_idx, _, _ = _training_shortlist(X_train, y_train, shortlist_size)
        return rng.choice(shortlist_idx, size=k, replace=False)

    detailed_results = {
        "QAOA-selected": _cross_validated_selector_auc(
            embeddings, y_encoded, qaoa_selector, k=k, random_state=random_state
        ),
        "Classical top-K (MI only)": _cross_validated_selector_auc(
            embeddings, y_encoded, classical_selector, k=k, random_state=random_state
        ),
        "Full embedding (baseline)": _cross_validated_selector_auc(
            embeddings, y_encoded,
            lambda X_train, _y_train, _fold: np.arange(X_train.shape[1]),
            k=embeddings.shape[1], random_state=random_state,
        ),
    }

    random_scores = []
    random_fold_scores = []
    for _ in range(n_random_draws):
        random_result = _cross_validated_selector_auc(
            embeddings, y_encoded, random_selector, k=k, random_state=random_state
        )
        random_scores.append(random_result[0])
        random_fold_scores.append(random_result[2])
    results = {
        name: (values[0], values[1])
        for name, values in detailed_results.items()
    }
    results["Random K (avg of draws)"] = (
        np.mean(random_scores), np.std(random_scores)
    )
    fold_scores = {
        name: values[2] for name, values in detailed_results.items()
    }
    fold_scores["Random K (avg of draws)"] = np.mean(random_fold_scores, axis=0)

    print("\n" + "=" * 60)
    print(f"{'Method':<30}{'Mean AUC':>12}{'Std':>10}")
    print("=" * 60)
    for name, (mean_auc, std_auc) in results.items():
        print(f"{name:<30}{mean_auc:>12.3f}{std_auc:>10.3f}")
    print("=" * 60)
    print(
        "Read this as: QAOA is only 'working' if its row beats BOTH "
        "'Classical top-K' and 'Random K' by more than roughly one std. "
        "If QAOA ties Classical top-K, the redundancy term isn't adding "
        "value here yet. If it ties Random K, something's wrong upstream "
        "(check the QUBO weights or that COBYLA actually converged)."
    )
    return results, fold_scores


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fair CNN embedding feature-selection comparison")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to Model.h5")
    parser.add_argument(
        "--data-root", default=DATA_ROOT,
        help="Parent directory containing cloudy, desert, green area, and water folders",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-draws", type=int, default=10)
    parser.add_argument("--qaoa-maxiter", type=int, default=QAOA_MAXITER)
    parser.add_argument(
        "--shortlist-size", type=int, default=SHORTLIST_SIZE,
        help="Number of candidate embedding dimensions passed to QAOA",
    )
    parser.add_argument(
        "--k", type=int, default=K,
        help="Number of dimensions selected from the shortlist",
    )
    parser.add_argument(
        "--qaoa-reps", type=int, default=QAOA_REPS,
        help="QAOA circuit depth",
    )
    parser.add_argument("--output-csv", default="qaoa_comparison.csv")
    parser.add_argument(
        "--quick", action="store_true",
        help="Use 3 random draws and 50 QAOA iterations for a smoke test",
    )
    args = parser.parse_args()

    full_model, feature_extractor = build_feature_extractor(args.model)
    input_size = full_model.input_shape[1:3]  # (height, width)

    embeddings, y_encoded, le = load_dataset_embeddings(
        feature_extractor, input_size, data_root=args.data_root,
        batch_size=args.batch_size,
    )

    random_draws = 3 if args.quick else args.random_draws
    qaoa_maxiter = 50 if args.quick else args.qaoa_maxiter
    if not 1 <= args.k <= args.shortlist_size:
        parser.error("--k must be between 1 and --shortlist-size")
    results, fold_scores = run_full_comparison(
        embeddings, y_encoded, n_random_draws=random_draws,
        k=args.k, shortlist_size=args.shortlist_size,
        qaoa_maxiter=qaoa_maxiter, qaoa_reps=args.qaoa_reps,
    )
    import csv
    method_names = list(results)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["method", "mean_auc", "std_auc"] + [
            f"fold_{index}" for index in range(1, len(next(iter(fold_scores.values()))) + 1)
        ])
        for name in method_names:
            mean_auc, std_auc = results[name]
            writer.writerow([name, mean_auc, std_auc, *fold_scores[name]])

    import matplotlib.pyplot as plt
    means = [results[name][0] for name in method_names]
    stds = [results[name][1] for name in method_names]
    plt.figure(figsize=(10, 5))
    plt.bar(method_names, means, yerr=stds, capsize=5, color=["#4063D8", "#389826", "#CB3C33", "#9558B2"])
    plt.ylabel("Mean ROC AUC")
    plt.title("CNN embedding feature-selection comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(args.output_csv.replace(".csv", "_bar.png"), dpi=180)
    plt.show()

    plt.figure(figsize=(10, 5))
    for name in method_names:
        plt.plot(range(1, len(fold_scores[name]) + 1), fold_scores[name], marker="o", label=name)
    plt.xlabel("Cross-validation fold")
    plt.ylabel("ROC AUC")
    plt.title("Per-fold ROC AUC")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_csv.replace(".csv", "_folds.png"), dpi=180)
    plt.show()

    qaoa_mean = results["QAOA-selected"][0]
    deltas = [results[name][0] - qaoa_mean for name in method_names]
    plt.figure(figsize=(10, 5))
    plt.bar(method_names, deltas, color="#E69F00")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Mean AUC minus QAOA AUC")
    plt.title("Performance difference relative to QAOA-selected features")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(args.output_csv.replace(".csv", "_delta.png"), dpi=180)
    plt.show()
    print(f"Saved comparison table and plots beside {args.output_csv}")
