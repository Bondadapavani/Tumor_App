import os
import cv2
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- MRI VALIDATION ----------
def is_mri_image(path):
    img = cv2.imread(path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)

    # MRI images usually have medium contrast
    return 400 < variance < 6000

# ---------- DEMO PREDICTION ----------
def predict_tumor(path):
    img = cv2.imread(path, 0)
    mean_intensity = np.mean(img)

    if mean_intensity < 70:
        return "Glioma", 87
    elif mean_intensity < 100:
        return "Meningioma", 83
    elif mean_intensity < 130:
        return "Pituitary", 85
    else:
        return "No Tumor", 92

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

            if not is_mri_image(path):
                message = "❌ INVALID IMAGE - PLEASE UPLOAD MRI SCAN ONLY"
            else:
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
