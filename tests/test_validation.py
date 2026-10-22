"""Tests for input validation and 400 error handling on POST /predict."""

import json


def test_validation_missing_feature(client, sample_wine_payload):
    """Missing a required feature returns 400 with field-level error detail."""
    payload = dict(sample_wine_payload)
    del payload["alcohol"]

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()

    assert "details" in data
    assert any("alcohol" in d.get("field", "") for d in data["details"])


def test_validation_string_instead_of_number(client, sample_wine_payload):
    """String instead of numeric feature returns 400 with field-level error detail."""
    payload = dict(sample_wine_payload)
    payload["alcohol"] = "invalid_string"

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()

    assert "details" in data
    assert any("alcohol" in d.get("field", "") for d in data["details"])


def test_validation_extra_field_rejected(client, sample_wine_payload):
    """Extra unpermitted field is rejected with 400 per pydantic extra='forbid'."""
    payload = dict(sample_wine_payload)
    payload["vintage_year"] = 1998

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()

    assert "details" in data
    assert any("vintage_year" in d.get("field", "") for d in data["details"])


def test_validation_nan_value_rejected(client, sample_wine_payload):
    """NaN numeric values are rejected with 400."""
    payload = dict(sample_wine_payload)
    # NaN in JSON can be sent as raw float('nan') or serialized null/NaN
    payload["alcohol"] = float("nan")

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "details" in data


def test_validation_infinite_value_rejected(client, sample_wine_payload):
    """Infinite numeric values are rejected with 400."""
    payload = dict(sample_wine_payload)
    payload["alcohol"] = float("inf")

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "details" in data


def test_validation_non_json_content_type(client):
    """Non-JSON Content-Type returns 400 Bad Request."""
    response = client.post("/predict", data="alcohol=13.2", content_type="text/plain")
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_validation_non_object_json_body(client):
    """JSON list instead of object returns 400 Bad Request."""
    response = client.post("/predict", json=[14.23, 1.71, 2.43])
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_validation_empty_body(client):
    """Empty JSON object returns 400 with missing field details."""
    response = client.post("/predict", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert "details" in data
    assert len(data["details"]) == 13
