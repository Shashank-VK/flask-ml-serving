"""Tests for GET /model-info endpoint."""


def test_model_info_returns_200(client):
    """GET /model-info must return 200 OK."""
    response = client.get("/model-info")
    assert response.status_code == 200


def test_model_info_matches_metrics_json(client, metrics_data):
    """GET /model-info response must match the metrics.json artifact exactly."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.get_json()
    assert data == metrics_data


def test_model_info_required_fields_present(client):
    """GET /model-info must contain algorithm, test_accuracy, features, and classes."""
    response = client.get("/model-info")
    data = response.get_json()
    assert "algorithm" in data
    assert "test_accuracy" in data
    assert "features" in data
    assert "classes" in data
    assert isinstance(data["algorithm"], str)
    assert isinstance(data["test_accuracy"], float)
    assert isinstance(data["features"], list)
    assert isinstance(data["classes"], list)


def test_model_info_features_and_classes_content(client):
    """Validate feature count and class labels in /model-info."""
    response = client.get("/model-info")
    data = response.get_json()
    assert len(data["features"]) == 13
    assert len(data["classes"]) == 3
    assert data["classes"] == ["class_0", "class_1", "class_2"]
    assert "od280/od315_of_diluted_wines" in data["features"]
    assert 0.0 <= data["test_accuracy"] <= 1.0
