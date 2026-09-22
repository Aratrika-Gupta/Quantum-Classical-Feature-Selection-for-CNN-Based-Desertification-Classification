"""Leakage-safe CNN retraining and QAOA feature-selection evaluation."""

import argparse
import csv
import os
import random
import time

import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import qaoa_feature_selection as pipeline


def collect_images(data_root):
    paths, labels = [], []
    for relative_folder, label in pipeline.LABELS.items():
        folder = os.path.join(data_root, relative_folder)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Dataset folder not found: {folder}")
        for filename in sorted(os.listdir(folder)):
            path = os.path.join(folder, filename)
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, ValueError):
                continue
            paths.append(path)
            labels.append(label)
    if not paths:
        raise ValueError(f"No readable images found below {data_root}.")
    return np.asarray(paths), np.asarray(labels)


class ImageSequence(Sequence):
    def __init__(self, paths, labels, input_size, batch_size, shuffle, seed):
        super().__init__()
        self.paths = paths
        self.labels = labels
        self.input_size = tuple(input_size)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        images = []
        for item in indices:
            with Image.open(self.paths[item]) as image:
                images.append(
                    np.asarray(
                        image.convert("RGB").resize(self.input_size),
                        dtype=np.float32,
                    ) / 255.0
                )
        return np.asarray(images), self.labels[indices]

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


def extract_embeddings(extractor, paths, labels, input_size, batch_size):
    sequence = ImageSequence(
        paths, labels, input_size, batch_size, shuffle=False, seed=0
    )
    embeddings = extractor.predict(sequence, verbose=1)
    if embeddings.ndim != 2:
        raise ValueError(
            f"Embedding layer must produce a 2-D array, got {embeddings.shape}."
        )
    return embeddings


def fit_classifier(X_train, y_train, X_test, y_test, seed):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(X_test)
    classifier = RandomForestClassifier(n_estimators=200, random_state=seed)
    classifier.fit(train_scaled, y_train)
    probabilities = classifier.predict_proba(test_scaled)
    predictions = classifier.predict(test_scaled)
    auc = roc_auc_score(
        y_test,
        probabilities,
        multi_class="ovr",
        labels=np.arange(len(np.unique(y_train))),
    )
    return auc, accuracy_score(y_test, predictions)


def main():
    parser = argparse.ArgumentParser(
        description="Retrain CNN and evaluate QAOA feature selection on a held-out test set"
    )
    parser.add_argument("--model", default="Model.h5")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shortlist-size", type=int, default=10)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--qaoa-reps", type=int, default=1)
    parser.add_argument("--qaoa-maxiter", type=int, default=50)
    parser.add_argument(
        "--random-draws", type=int, default=1,
        help="Number of QAOA and random-K selections to evaluate",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-model", default="Model_holdout_retrained.h5")
    parser.add_argument("--output-csv", default="holdout_feature_selection.csv")
    args = parser.parse_args()

    if not 0 < args.test_size < 1 or not 0 < args.validation_size < 1:
        parser.error("test and validation sizes must be between 0 and 1")
    if not 1 <= args.k <= args.shortlist_size:
        parser.error("--k must be between 1 and --shortlist-size")
    if args.random_draws < 1:
        parser.error("--random-draws must be at least 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    paths, label_names = collect_images(args.data_root)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(label_names)
    train_paths, test_paths, y_train, y_test = train_test_split(
        paths, labels, test_size=args.test_size, stratify=labels,
        random_state=args.seed,
    )
    validation_fraction = args.validation_size / (1.0 - args.test_size)
    train_paths, validation_paths, y_train, y_validation = train_test_split(
        train_paths, y_train, test_size=validation_fraction,
        stratify=y_train, random_state=args.seed,
    )
    print(
        f"Split: train={len(train_paths)}, validation={len(validation_paths)}, "
        f"test={len(test_paths)}"
    )

    from tensorflow.keras import Model
    from tensorflow.keras.models import clone_model, load_model

    source_model = load_model(args.model, compile=False)
    model = clone_model(source_model)
    model.build(source_model.input_shape)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    input_size = source_model.input_shape[1:3]
    train_sequence = ImageSequence(
        train_paths, y_train, input_size, args.batch_size, shuffle=True, seed=args.seed
    )
    validation_sequence = ImageSequence(
        validation_paths, y_validation, input_size, args.batch_size,
        shuffle=False, seed=args.seed,
    )
    started = time.perf_counter()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] RETRAIN START")
    model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=args.epochs,
        verbose=1,
    )
    model.save(args.output_model)
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] RETRAIN END after "
        f"{time.perf_counter() - started:.1f}s; saved {args.output_model}"
    )

    extractor = Model(
        inputs=model.inputs,
        outputs=model.layers[pipeline.EMBEDDING_LAYER_INDEX].output,
    )
    X_train = extract_embeddings(
        extractor, train_paths, y_train, input_size, args.batch_size
    )
    X_test = extract_embeddings(
        extractor, test_paths, y_test, input_size, args.batch_size
    )

    mi_scores = mutual_info_classif(X_train, y_train, random_state=args.seed)
    shortlist_idx = np.argsort(mi_scores)[::-1][:args.shortlist_size]
    X_shortlist = X_train[:, shortlist_idx]
    relevance = mi_scores[shortlist_idx]
    relevance = (relevance - relevance.min()) / (
        relevance.max() - relevance.min() + 1e-9
    )
    redundancy = np.nan_to_num(np.abs(np.corrcoef(X_shortlist.T)), nan=0.0)
    np.fill_diagonal(redundancy, 0.0)
    classical_idx = shortlist_idx[:args.k]
    rng = np.random.default_rng(args.seed)
    evaluation_rows = []
    qaoa_results = []
    for draw in range(args.random_draws):
        selected_mask = pipeline.qaoa_select_features(
            relevance, redundancy, k=args.k, reps=args.qaoa_reps,
            maxiter=args.qaoa_maxiter, seed=args.seed + draw,
        )
        qaoa_results.append(shortlist_idx[selected_mask])
        random_idx = rng.choice(shortlist_idx, size=args.k, replace=False)
        for name, indices in (
            ("QAOA-selected", qaoa_results[-1]),
            ("Random K", random_idx),
        ):
            auc, accuracy = fit_classifier(
                X_train[:, indices], y_train, X_test[:, indices], y_test,
                args.seed + draw,
            )
            evaluation_rows.append((name, draw + 1, auc, accuracy))

    for name, indices in (
        ("Classical top-K (MI only)", classical_idx),
        ("Full embedding (baseline)", np.arange(X_train.shape[1])),
    ):
        auc, accuracy = fit_classifier(
            X_train[:, indices], y_train, X_test[:, indices], y_test, args.seed
        )
        evaluation_rows.append((name, 1, auc, accuracy))

    print("\nHeld-out test results (test images were never used for fitting):")
    print(f"{'Method':<30}{'AUC':>15}{'Accuracy':>15}")
    summary = {}
    for name in ("QAOA-selected", "Random K", "Classical top-K (MI only)",
                 "Full embedding (baseline)"):
        values = [(auc, accuracy) for row_name, _, auc, accuracy in evaluation_rows
                  if row_name == name]
        aucs, accuracies = np.asarray(values).T
        summary[name] = (aucs.mean(), aucs.std(), accuracies.mean(), accuracies.std())
        print(
            f"{name:<30}{aucs.mean():>7.3f} +/- {aucs.std():<5.3f}"
            f"{accuracies.mean():>7.3f} +/- {accuracies.std():<5.3f}"
        )

    with open(args.output_csv, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["method", "draw", "test_auc", "test_accuracy"])
        writer.writerows(evaluation_rows)
        writer.writerow([])
        writer.writerow(["method", "mean_auc", "std_auc", "mean_accuracy",
                         "std_accuracy"])
        for name, values in summary.items():
            writer.writerow([name, *values])
    print(f"Saved held-out results to {args.output_csv}")


if __name__ == "__main__":
    main()
