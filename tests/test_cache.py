"""Tests for prediction caching behavior."""


def test_cache_flag_flips_on_repeat_call(client, sample_wine_payload):
    """First call must return cached=False, second identical call must return cached=True."""
    res1 = client.post("/predict", json=sample_wine_payload)
    assert res1.status_code == 200
    data1 = res1.get_json()
    assert data1["cached"] is False

    res2 = client.post("/predict", json=sample_wine_payload)
    assert res2.status_code == 200
    data2 = res2.get_json()
    assert data2["cached"] is True

    # Validate predictions match
    assert data1["prediction"] == data2["prediction"]
    assert data1["class_index"] == data2["class_index"]
    assert data1["probabilities"] == data2["probabilities"]


def test_cache_distinct_inputs_independent(client, sample_wine_payload, alternative_wine_payload):
    """Different feature payloads are cached independently."""
    # Call A
    res_a1 = client.post("/predict", json=sample_wine_payload).get_json()
    assert res_a1["cached"] is False

    # Call B (different features)
    res_b1 = client.post("/predict", json=alternative_wine_payload).get_json()
    assert res_b1["cached"] is False

    # Repeat A -> hit
    res_a2 = client.post("/predict", json=sample_wine_payload).get_json()
    assert res_a2["cached"] is True

    # Repeat B -> hit
    res_b2 = client.post("/predict", json=alternative_wine_payload).get_json()
    assert res_b2["cached"] is True


def test_cache_clear_resets_cached_flag(app, client, sample_wine_payload):
    """Clearing the cache resets the cached flag to False on subsequent call."""
    res1 = client.post("/predict", json=sample_wine_payload).get_json()
    assert res1["cached"] is False

    res2 = client.post("/predict", json=sample_wine_payload).get_json()
    assert res2["cached"] is True

    # Clear cache
    app.extensions["prediction_cache"].clear()

    res3 = client.post("/predict", json=sample_wine_payload).get_json()
    assert res3["cached"] is False
