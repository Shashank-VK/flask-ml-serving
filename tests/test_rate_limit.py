"""Tests for rate limiting behavior and 429 responses."""

from app import create_app
from app.extensions import limiter


def test_rate_limit_exceeded_produces_429(sample_wine_payload):
    """Configuring a tiny rate limit must return 429 once threshold is exceeded."""
    app = create_app({
        "TESTING": True,
        "RATELIMIT_DEFAULT": "3 per minute",
        "RATELIMIT_STORAGE_URI": "memory://",
    })
    limiter.reset()
    client = app.test_client()

    # Requests 1 to 3 should succeed
    for _ in range(3):
        res = client.post("/predict", json=sample_wine_payload)
        assert res.status_code == 200

    # Request 4 exceeds the 3/min limit and must produce 429
    res4 = client.post("/predict", json=sample_wine_payload)
    assert res4.status_code == 429
    data = res4.get_json()
    assert data["error"] == "Rate limit exceeded"
    assert "3 per 1 minute" in data["message"]
    limiter.reset()


def test_rate_limit_response_json_format():
    """Exceeded rate limit response contains expected JSON structure."""
    app = create_app({
        "TESTING": True,
        "RATELIMIT_DEFAULT": "1 per minute",
        "RATELIMIT_STORAGE_URI": "memory://",
    })
    limiter.reset()
    client = app.test_client()

    res1 = client.get("/model-info")
    assert res1.status_code == 200

    res2 = client.get("/model-info")
    assert res2.status_code == 429
    data = res2.get_json()
    assert "error" in data
    assert "message" in data
    assert data["error"] == "Rate limit exceeded"
    limiter.reset()
