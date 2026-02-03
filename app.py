from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import numpy as np
import cv2
import traceback
import os

app = Flask(__name__)
CORS(app)

# ---- PRELOAD MODEL ONCE ----
print("🚀 Loading DeepFace gender model...")
gender_model = DeepFace.build_model("Gender")
print("✅ Gender model loaded")

def classify_gender(image_bytes, min_confidence=60):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image data")

    result = DeepFace.analyze(
        img,
        actions=["gender"],
        detector_backend="opencv",   # MUCH lighter
        enforce_detection=False,     # critical for stability
        align=False
    )

    gender_scores = result[0]["gender"] if isinstance(result, list) else result["gender"]

    predicted_gender = max(gender_scores, key=gender_scores.get)
    confidence = float(gender_scores[predicted_gender])

    if confidence < min_confidence:
        raise ValueError(f"Low confidence ({confidence:.2f}%)")

    return predicted_gender, confidence


@app.route("/", methods=["GET"])
def home():
    return {"status": "ok"}


@app.route("/predict_gender", methods=["POST"])
def predict_gender():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        image_bytes = request.files["image"].read()
        gender, confidence = classify_gender(image_bytes)
        return jsonify({"gender": gender, "confidence": confidence})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False,threaded=False)

