# Wine Quality Inference API (`flask-ml-serving`)

A Flask inference API demonstrating production ML model serving practices: reproducible offline training, versioned model artifacts, strict input validation with Pydantic v2, in-process prediction caching, configurable rate limiting with Flask-Limiter, and a comprehensive pytest suite.

---

## What It Is

This project serves an inference API for a multi-class classification model predicting wine cultivar classes from 13 chemical features. It showcases essential model serving patterns without unnecessary infrastructure overhead:

- **Isolated & Reproducible Training**: Dedicated `train.py` script producing serialized model weights (`model.pkl`) and training metadata (`metrics.json`).
- **Committed Pre-trained Artifacts**: Artifacts generated during build are bundled under `app/artifacts/` so the service runs immediately out of the box.
- **Strict Input Validation**: Request payloads are parsed and validated using Pydantic v2 schemas (`extra='forbid'`, finite float checks, field-level 400 error reporting).
- **In-Process Prediction Caching**: Exact feature vectors are converted to canonical tuples and cached in an in-memory LRU cache (`cached: true` on repeat calls).
- **Configurable Rate Limiting**: IP-based rate limiting via Flask-Limiter with in-memory backend, configurable through environment variables, returning 429 JSON responses.
- **Automated Test Coverage**: 24 tests covering every endpoint, input validation edge case, cache behavior, determinism, and rate limiting.

---

## Dataset (Demonstration Framing)

This project uses the Wine recognition dataset bundled directly with scikit-learn (`sklearn.datasets.load_wine`).

- **Dataset Source**: `sklearn.datasets.load_wine()` (no external download required)
- **Total Samples**: 178
- **Number of Features**: 13 continuous chemical measurements
- **Classes**: 3 wine cultivars (`class_0`, `class_1`, `class_2`)
- **Framing Notice**: This is a benchmark demonstration dataset intended to prove ML API architecture, serving ergonomics, and testing patterns—not a production agricultural decision support system.

### The 13 Features
1. `alcohol`
2. `malic_acid`
3. `ash`
4. `alcalinity_of_ash`
5. `magnesium`
6. `total_phenols`
7. `flavanoids`
8. `nonflavanoid_phenols`
9. `proanthocyanins`
10. `color_intensity`
11. `hue`
12. `od280/od315_of_diluted_wines`
13. `proline`

---

## Model Training & Metrics

The classifier was trained using `train.py` with the following configuration:

- **Algorithm**: `RandomForestClassifier` (`n_estimators=100`, `random_state=42`)
- **Data Split**: Stratified 80/20 train/test split (`random_state=42`, `stratify=y`)
  - Train set: 142 samples
  - Test set: 36 samples
- **Scikit-Learn Version**: `1.5.0`
- **Real Test Accuracy (from build run)**: `1.0` (100.0% accuracy on the 36-sample test partition, as recorded in `app/artifacts/metrics.json`)
- **Trained At**: `2026-10-22T10:18:08.716942+00:00`

---

## Setup, Run & Test

### 1. Prerequisites
- Python 3.12+
- `virtualenv` or Python standard library `venv`

### 2. Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install locked dependencies
pip install -r requirements.txt
```

### 3. (Optional) Re-train Model Artifacts
The repository includes pre-built artifacts in `app/artifacts/`. To retrain and regenerate them:
```bash
python train.py
```

### 4. Run the API Server
```bash
# Run with WSGI entry point
python wsgi.py
```
The server will start listening at `http://0.0.0.0:5000`.

### 5. Run the Test Suite
```bash
python -m pytest
```
The test suite consists of 24 tests across unit, integration, validation, caching, and rate limiting modules.

---

## API Endpoints

| Method | Endpoint | Description | Status Codes |
|---|---|---|---|
| `GET` | `/health` | Service health status. Exempt from rate limiting. | `200` |
| `GET` | `/model-info` | Metadata from `metrics.json` (algorithm, accuracy, classes, features). | `200`, `429` |
| `POST` | `/predict` | Model inference on 13 wine chemical features. | `200`, `400`, `429` |

---

## Example Usage

### Health Check
```bash
curl -X GET http://localhost:5000/health
```
**Response:**
```json
{
  "status": "healthy"
}
```

### Model Info
```bash
curl -X GET http://localhost:5000/model-info
```
**Response:**
```json
{
  "algorithm": "RandomForestClassifier",
  "classes": [
    "class_0",
    "class_1",
    "class_2"
  ],
  "features": [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline"
  ],
  "n_features": 13,
  "n_samples": 178,
  "n_test_samples": 36,
  "n_train_samples": 142,
  "random_state": 42,
  "sklearn_version": "1.5.0",
  "test_accuracy": 1.0,
  "test_size": 0.2,
  "trained_at": "2026-10-22T10:18:08.716942+00:00"
}
```

### Predict (cURL Example)
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "proline": 1065.0
  }'
```

**Response (First Call — Cache Miss):**
```json
{
  "cached": false,
  "class_index": 0,
  "prediction": "class_0",
  "probabilities": {
    "class_0": 1.0,
    "class_1": 0.0,
    "class_2": 0.0
  }
}
```

**Response (Second Call with Identical Payload — Cache Hit):**
```json
{
  "cached": true,
  "class_index": 0,
  "prediction": "class_0",
  "probabilities": {
    "class_0": 1.0,
    "class_1": 0.0,
    "class_2": 0.0
  }
}
```

### Validation Error Example
If a feature is missing, extra fields are supplied, or a type is incorrect:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "alcohol": "not-a-number",
    "extra_field": 123
  }'
```
**Response (400 Bad Request):**
```json
{
  "details": [
    {
      "field": "alcohol",
      "message": "Input should be a valid number, unable to parse string as a number",
      "type": "float_parsing"
    },
    {
      "field": "malic_acid",
      "message": "Field required",
      "type": "missing"
    },
    {
      "field": "extra_field",
      "message": "Extra inputs are not permitted",
      "type": "extra_forbidden"
    }
  ],
  "error": "Validation Error",
  "message": "Input features failed validation"
}
```

---

## Cache & Rate-Limit Configuration

### Prediction Cache
- **Mechanism**: In-process thread-safe LRU cache implemented via `collections.OrderedDict` protected by a `threading.Lock`.
- **Keying**: Canonical 13-element tuple of float feature values in the exact order declared by `metrics.json["features"]`.
- **Behavior**:
  - The first request computes inference with scikit-learn and returns `"cached": false`.
  - Subsequent requests with identical features return stored inferences with `"cached": true`.
- **Configuration**:
  - `CACHE_MAX_SIZE`: Maximum number of entries kept in cache before LRU eviction (default: `1024`).

### Rate Limiting
- **Mechanism**: Flask-Limiter extension using in-memory storage (`memory://`) and client IP resolution (`get_remote_address`).
- **Endpoint Exemption**: The `/health` endpoint is decorated with `@limiter.exempt` to prevent monitoring health checks from being throttled.
- **Exceeded Threshold**: Exceeding the rate limit returns a `429 Too Many Requests` status code with JSON body:
  ```json
  {
    "error": "Rate limit exceeded",
    "message": "60 per 1 minute"
  }
  ```
- **Configuration**:
  - `RATELIMIT_DEFAULT`: Default rate limit string (default: `"60 per minute"`). Can be set via environment variable:
    ```bash
    export RATELIMIT_DEFAULT="120 per minute"
    python wsgi.py
    ```
