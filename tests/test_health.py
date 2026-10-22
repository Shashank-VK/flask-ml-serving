"""Tests for GET /health endpoint."""

from app import create_app


def test_health_returns_200_and_healthy(client):
    """GET /health must return 200 with status healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data == {"status": "healthy"}


def test_health_exempt_from_rate_limit():
    """GET /health must be exempt from rate limiting even under heavy traffic."""
    app = create_app({
        "TESTING": True,
        "RATELIMIT_DEFAULT": "2 per minute",
        "RATELIMIT_STORAGE_URI": "memory://",
    })
    client = app.test_client()

    for _ in range(5):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.get_json() == {"status": "healthy"}
