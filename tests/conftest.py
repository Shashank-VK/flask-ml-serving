"""Pytest configuration and shared fixtures."""

import json
import os
import pytest

from app import create_app
from app.extensions import limiter


@pytest.fixture
def app():
    """Create application instance configured for testing."""
    test_app = create_app({
        "TESTING": True,
        "RATELIMIT_DEFAULT": "1000 per minute",
        "RATELIMIT_STORAGE_URI": "memory://",
    })
    limiter.reset()
    test_app.extensions["prediction_cache"].clear()
    yield test_app
    test_app.extensions["prediction_cache"].clear()
    limiter.reset()


@pytest.fixture
def client(app):
    """Test client for HTTP requests."""
    return app.test_client()


@pytest.fixture
def sample_wine_payload():
    """Valid wine feature payload corresponding to class_0."""
    return {
        "alcohol": 14.23,
        "malic_acid": 1.71,
        "ash": 2.43,
        "alcalinity_of_ash": 15.6,
        "magnesium": 127.0,
        "total_phenols": 2.8,
        "flavanoids": 3.06,
        "nonflavanoid_phenols": 0.28,
        "proanthocyanins": 2.29,
        "color_intensity": 5.64,
        "hue": 1.04,
        "od280/od315_of_diluted_wines": 3.92,
        "proline": 1065.0,
    }


@pytest.fixture
def alternative_wine_payload():
    """Valid wine feature payload corresponding to class_1."""
    return {
        "alcohol": 12.37,
        "malic_acid": 0.94,
        "ash": 1.36,
        "alcalinity_of_ash": 10.6,
        "magnesium": 88.0,
        "total_phenols": 1.98,
        "flavanoids": 0.57,
        "nonflavanoid_phenols": 0.28,
        "proanthocyanins": 0.42,
        "color_intensity": 1.95,
        "hue": 1.05,
        "od280/od315_of_diluted_wines": 1.82,
        "proline": 520.0,
    }


@pytest.fixture
def sample_wine_class_2():
    """Valid wine feature payload corresponding to class_2."""
    return {
        "alcohol": 12.86,
        "malic_acid": 1.35,
        "ash": 2.32,
        "alcalinity_of_ash": 18.0,
        "magnesium": 122.0,
        "total_phenols": 1.51,
        "flavanoids": 1.25,
        "nonflavanoid_phenols": 0.21,
        "proanthocyanins": 0.94,
        "color_intensity": 4.1,
        "hue": 0.76,
        "od280/od315_of_diluted_wines": 1.29,
        "proline": 630.0,
    }


@pytest.fixture
def metrics_data():
    """Load actual metrics.json from disk."""
    metrics_path = os.path.join(
        os.path.dirname(__file__), "..", "app", "artifacts", "metrics.json"
    )
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)
