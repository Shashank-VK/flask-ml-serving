"""API routes and endpoints for wine inference service."""

from flask import Blueprint, current_app, jsonify, request
from pydantic import ValidationError

from app.extensions import limiter
from app.schemas import WineFeatures

api_bp = Blueprint("api", __name__)


@api_bp.route("/health", methods=["GET"])
@limiter.exempt
def health():
    """Health check endpoint.

    Returns 200 OK with status healthy. Exempt from rate limiting.
    """
    return jsonify({"status": "healthy"}), 200


@api_bp.route("/model-info", methods=["GET"])
def model_info():
    """Retrieve metadata about the currently deployed model.

    Returns metrics.json contents including algorithm, test_accuracy, features, and classes.
    """
    model_service = current_app.extensions["model_service"]
    return jsonify(model_service.metrics), 200


@api_bp.route("/predict", methods=["POST"])
def predict():
    """Inference endpoint.

    Expects a JSON body with 13 wine chemical features.
    Returns 200 with prediction, class_index, probabilities, and cached flag.
    Returns 400 with field-level error details on validation failure.
    """
    if not request.is_json:
        return jsonify({
            "error": "Bad Request",
            "message": "Content-Type must be application/json",
            "details": [{"field": "body", "message": "Expected JSON payload", "type": "type_error"}]
        }), 400

    raw_json = request.get_json(silent=True)
    if raw_json is None or not isinstance(raw_json, dict):
        return jsonify({
            "error": "Bad Request",
            "message": "Request body must be a valid JSON object",
            "details": [{"field": "body", "message": "Expected a JSON object mapping feature names to numbers", "type": "type_error"}]
        }), 400

    # Pydantic v2 validation
    try:
        features = WineFeatures.model_validate(raw_json)
    except ValidationError as e:
        field_details = []
        for err in e.errors():
            loc_parts = err.get("loc", ())
            field_name = ".".join(str(p) for p in loc_parts) if loc_parts else "body"
            field_details.append({
                "field": field_name,
                "message": err.get("msg", "Invalid value"),
                "type": err.get("type", "value_error"),
            })
        return jsonify({
            "error": "Validation Error",
            "message": "Input features failed validation",
            "details": field_details
        }), 400

    cache = current_app.extensions["prediction_cache"]
    model_service = current_app.extensions["model_service"]

    cache_key = features.to_cache_key()
    cached_entry = cache.get(cache_key)

    if cached_entry is not None:
        response_data = dict(cached_entry)
        response_data["cached"] = True
        return jsonify(response_data), 200

    # Cache miss: compute prediction
    prediction_result = model_service.predict(features.to_feature_list())
    cache.set(cache_key, prediction_result)

    response_data = dict(prediction_result)
    response_data["cached"] = False
    return jsonify(response_data), 200
