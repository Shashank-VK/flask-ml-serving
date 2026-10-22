"""Application factory for Flask ML Serving."""

import os
from typing import Any, Dict, Optional

from flask import Flask, jsonify

from app.cache import PredictionCache
from app.config import Config
from app.extensions import limiter
from app.model import ModelService
from app.routes import api_bp


def create_app(test_config: Optional[Dict[str, Any]] = None) -> Flask:
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)
    app.config.from_object(Config)

    if test_config:
        app.config.update(test_config)

    # Initialize rate limiter
    limiter.init_app(app)

    # Initialize model service and prediction cache
    artifacts_dir = app.config.get("ARTIFACTS_DIR")
    if not os.path.isabs(artifacts_dir):
        artifacts_dir = os.path.join(app.root_path, "..", artifacts_dir)

    model_service = ModelService(artifacts_dir=artifacts_dir)
    prediction_cache = PredictionCache(maxsize=app.config.get("CACHE_MAX_SIZE", 1024))

    app.extensions["model_service"] = model_service
    app.extensions["prediction_cache"] = prediction_cache

    # Register blueprints
    app.register_blueprint(api_bp)

    # Error handlers
    @app.errorhandler(429)
    def ratelimit_handler(e):
        return jsonify({
            "error": "Rate limit exceeded",
            "message": str(e.description if hasattr(e, "description") else e),
        }), 429

    @app.errorhandler(404)
    def not_found_handler(e):
        return jsonify({
            "error": "Not Found",
            "message": "The requested resource was not found on this server.",
        }), 404

    @app.errorhandler(405)
    def method_not_allowed_handler(e):
        return jsonify({
            "error": "Method Not Allowed",
            "message": "The method is not allowed for the requested URL.",
        }), 405

    @app.errorhandler(500)
    def internal_error_handler(e):
        return jsonify({
            "error": "Internal Server Error",
            "message": "An internal error occurred while processing the request.",
        }), 500

    return app
