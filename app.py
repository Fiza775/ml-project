from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Serve index.html from the same directory as app.py
app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)

# ─────────────────────────────────────────────
# Load saved model & scaler on startup
# ─────────────────────────────────────────────
try:
    model  = joblib.load(os.path.join(BASE_DIR, "random_forest_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    print("✅  Model and scaler loaded successfully.")
except FileNotFoundError as e:
    raise RuntimeError(
        "Model files not found. Please run the notebook first to generate "
        "'random_forest_model.pkl' and 'scaler.pkl'."
    ) from e


# ─────────────────────────────────────────────
# Feature order (must match training columns)
# gender, age, hypertension, heart_disease,
# smoking_history, bmi, HbA1c_level,
# blood_glucose_level
# ─────────────────────────────────────────────
FEATURE_ORDER = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

# Label encoding maps (mirrors the notebook's LabelEncoder fit order)
GENDER_MAP  = {"Female": 0, "Male": 1, "Other": 2}
SMOKING_MAP = {
    "No Info": 0,
    "current": 1,
    "ever": 2,
    "former": 3,
    "never": 4,
    "not current": 5,
}


# ─────────────────────────────────────────────
# Serve the frontend
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """Open index.html in the browser at http://localhost:5000"""
    return send_from_directory(BASE_DIR, "index.html")


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Diabetes Prediction API is running."})


# ─────────────────────────────────────────────
# Prediction endpoint
# ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
    {
        "gender":              "Male" | "Female" | "Other",
        "age":                 <float>,
        "hypertension":        0 | 1,
        "heart_disease":       0 | 1,
        "smoking_history":     "never" | "former" | "current" | "ever" | "not current" | "No Info",
        "bmi":                 <float>,
        "HbA1c_level":         <float>,
        "blood_glucose_level": <int>
    }

    Returns:
    {
        "prediction":  0 | 1,
        "label":       "Diabetic" | "Non-Diabetic",
        "probability": { "non_diabetic": float, "diabetic": float }
    }
    """
    data = request.get_json(force=True)

    # Validate required keys
    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Encode categoricals
    gender_raw  = data["gender"]
    smoking_raw = data["smoking_history"]

    if gender_raw not in GENDER_MAP:
        return jsonify({"error": f"Invalid gender '{gender_raw}'. Choose from {list(GENDER_MAP.keys())}"}), 400
    if smoking_raw not in SMOKING_MAP:
        return jsonify({"error": f"Invalid smoking_history '{smoking_raw}'. Choose from {list(SMOKING_MAP.keys())}"}), 400

    gender_enc  = GENDER_MAP[gender_raw]
    smoking_enc = SMOKING_MAP[smoking_raw]

    # Build feature vector
    try:
        features = np.array([[
            gender_enc,
            float(data["age"]),
            int(data["hypertension"]),
            int(data["heart_disease"]),
            smoking_enc,
            float(data["bmi"]),
            float(data["HbA1c_level"]),
            float(data["blood_glucose_level"]),
        ]])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400

    # Scale → predict
    features_scaled = scaler.transform(features)
    prediction      = int(model.predict(features_scaled)[0])
    probabilities   = model.predict_proba(features_scaled)[0]

    return jsonify({
        "prediction": prediction,
        "label":      "Diabetic" if prediction == 1 else "Non-Diabetic",
        "probability": {
            "non_diabetic": round(float(probabilities[0]) * 100, 2),
            "diabetic":     round(float(probabilities[1]) * 100, 2),
        },
    })


# ─────────────────────────────────────────────
# Feature info endpoint
# ─────────────────────────────────────────────
@app.route("/features", methods=["GET"])
def features():
    """Returns the expected input fields and their allowed values."""
    return jsonify({
        "fields": [
            {"name": "gender",              "type": "categorical", "options": list(GENDER_MAP.keys())},
            {"name": "age",                 "type": "float",       "range": [0, 120]},
            {"name": "hypertension",        "type": "binary",      "options": [0, 1]},
            {"name": "heart_disease",       "type": "binary",      "options": [0, 1]},
            {"name": "smoking_history",     "type": "categorical", "options": list(SMOKING_MAP.keys())},
            {"name": "bmi",                 "type": "float",       "range": [10, 70]},
            {"name": "HbA1c_level",         "type": "float",       "range": [3.5, 9.0]},
            {"name": "blood_glucose_level", "type": "int",         "range": [80, 300]},
        ]
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
