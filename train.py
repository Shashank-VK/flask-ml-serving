"""Training script for Wine Quality Classifier.

Loads the bundled scikit-learn Wine dataset, performs a stratified
train/test split with a fixed random seed, trains a RandomForestClassifier,
and saves model.pkl and metrics.json into app/artifacts/.
"""

import json
import os
import pickle
from datetime import datetime, timezone

import sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train(artifacts_dir: str = "app/artifacts", random_state: int = 42, test_size: float = 0.2) -> dict:
    """Train the model and save artifacts."""
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Load Wine dataset
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = list(wine.feature_names)
    target_names = list(wine.target_names)

    # 2. Stratified train/test split with fixed seed
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 3. Train RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)

    # 4. Evaluate on test set
    y_pred = clf.predict(X_test)
    test_accuracy = float(accuracy_score(y_test, y_pred))

    # 5. Prepare metrics dictionary
    metrics = {
        "algorithm": clf.__class__.__name__,
        "test_accuracy": test_accuracy,
        "classes": target_names,
        "features": feature_names,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "random_state": random_state,
        "test_size": test_size,
        "n_samples": int(len(X)),
        "n_features": int(len(feature_names)),
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
    }

    # 6. Save model artifact
    model_path = os.path.join(artifacts_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    # 7. Save metrics artifact
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Loaded Wine dataset: {len(X)} samples, {len(feature_names)} features, {len(target_names)} classes.")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Algorithm: {clf.__class__.__name__} (random_state={random_state})")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")

    return metrics


if __name__ == "__main__":
    train()
