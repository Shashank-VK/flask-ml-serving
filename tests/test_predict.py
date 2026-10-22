"""Tests for POST /predict endpoint happy paths and model behavior."""

import pytest


def test_predict_happy_path(client, sample_wine_payload):
    """POST /predict with valid features returns 200 with required structure."""
    response = client.post("/predict", json=sample_wine_payload)
    assert response.status_code == 200
    data = response.get_json()

    assert "prediction" in data
    assert "class_index" in data
    assert "probabilities" in data
    assert "cached" in data

    assert isinstance(data["prediction"], str)
    assert isinstance(data["class_index"], int)
    assert isinstance(data["probabilities"], dict)
    assert isinstance(data["cached"], bool)
    assert data["prediction"] in ["class_0", "class_1", "class_2"]
    assert 0 <= data["class_index"] <= 2


def test_predict_probabilities_sum_to_one(client, sample_wine_payload):
    """Probabilities in /predict response must sum to ~1.0 with individual values in [0, 1]."""
    response = client.post("/predict", json=sample_wine_payload)
    assert response.status_code == 200
    data = response.get_json()

    probs = data["probabilities"]
    assert len(probs) == 3
    for class_name in ["class_0", "class_1", "class_2"]:
        assert class_name in probs
        assert 0.0 <= probs[class_name] <= 1.0

    total_prob = sum(probs.values())
    assert total_prob == pytest.approx(1.0, rel=1e-4)


def test_predict_class_consistency(client, sample_wine_payload):
    """Prediction label must match the class corresponding to class_index."""
    response = client.post("/predict", json=sample_wine_payload)
    data = response.get_json()
    assert data["prediction"] == f"class_{data['class_index']}"


def test_predict_determinism(client, sample_wine_payload):
    """Repeated calls with identical input produce deterministic predictions and probabilities."""
    responses = [client.post("/predict", json=sample_wine_payload).get_json() for _ in range(5)]
    first = responses[0]
    for res in responses[1:]:
        assert res["prediction"] == first["prediction"]
        assert res["class_index"] == first["class_index"]
        assert res["probabilities"] == first["probabilities"]


def test_predict_multiple_classes(client, sample_wine_payload, alternative_wine_payload, sample_wine_class_2):
    """The model can predict different classes for distinct wine samples."""
    r0 = client.post("/predict", json=sample_wine_payload).get_json()
    r1 = client.post("/predict", json=alternative_wine_payload).get_json()
    r2 = client.post("/predict", json=sample_wine_class_2).get_json()

    assert r0["prediction"] == "class_0"
    assert r1["prediction"] == "class_1"
    assert r2["prediction"] == "class_2"
