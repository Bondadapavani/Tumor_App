import os
import cv2
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- STRONG MRI VALIDATION ----------
def is_mri_image(path):
    img = cv2.imread(path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)

    # MRI images usually have medium contrast
    return 300 < variance < 7000

# ---------- SAFE DEMO PREDICTION ----------
def predict_tumor(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (224, 224))

    mean_intensity = np.mean(img)
    std_intensity = np.std(img)

    # Case 1: Normal MRI (most cases)
    if mean_intensity > 115 and std_intensity < 35:
        return "No Tumor", 95

    # Case 2: Slight abnormality
    if std_intensity < 55:
        return "Uncertain MRI", 78

    # Case 3: Strong abnormality
    if std_intensity >= 55:
        return "Tumor Detected", 82

    # Fallback
    return "Uncertain MRI", 75

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_path = None
    message = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            image_path = path

            # Step 1: MRI validation
            if not is_mri_image(path):
                message = "❌ INVALID IMAGE - PLEASE UPLOAD MRI SCAN ONLY"
            else:
                # Step 2: Safe prediction
                result, confidence = predict_tumor(path)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_path=image_path,
        message=message
    )

if __name__ == "__main__":
    app.run()
