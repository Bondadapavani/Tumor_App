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

    # 1. Check grayscale similarity
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img)
    color_diff = np.mean(np.abs(b - g)) + np.mean(np.abs(g - r))

    # 2. Edge structure
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # 3. Intensity variance
    variance = np.var(gray)

    # MRI characteristics
    if color_diff < 15 and edge_density > 0.02 and 500 < variance < 6000:
        return True
    else:
        return False

# ---------- FAKE AI PREDICTION ----------
def predict_tumor(path):
    img = cv2.imread(path, 0)
    mean = np.mean(img)

    if mean < 70:
        return "Glioma", 88
    elif mean < 100:
        return "Meningioma", 84
    elif mean < 130:
        return "Pituitary", 86
    else:
        return "No Tumor", 92

# ---------- ROUTE ----------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_url = None
    message = None

    if request.method == "POST":
        file = request.files.get("file")

        if file:
            filename = "mri.jpg"   # overwrite each time
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)

            image_url = "/static/uploads/" + filename

            # Strong MRI filter
            if not is_mri_image(save_path):
                message = "❌ Invalid Image. Please upload MRI scan only."
            else:
                result, confidence = predict_tumor(save_path)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_url=image_url,
        message=message
    )

if __name__ == "__main__":
    app.run()
