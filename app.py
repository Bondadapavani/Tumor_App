import os
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, request

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load ONNX model
session = ort.InferenceSession("brain_tumor.onnx")
classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# ---------- STRONG MRI VALIDATION ----------
def is_mri_image(path):
    img = cv2.imread(path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # MRI has medium contrast, low color
    variance = np.var(gray)
    b, g, r = cv2.split(img)
    color_var = np.var(b) + np.var(g) + np.var(r)

    # Edge structure (MRI has brain structures)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # These values work well in practice
    if 400 < variance < 6000 and color_var < 8000 and edge_density > 0.01:
        return True
    return False

# ---------- REAL AI PREDICTION ----------
def predict_tumor(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=(0, 1))

    inputs = {session.get_inputs()[0].name: img}
    outputs = session.run(None, inputs)[0][0]

    idx = np.argmax(outputs)
    max_prob = float(outputs[idx])
    confidence = round(max_prob * 100, 2)

    # Safety threshold
    if max_prob < 0.80:
        return "Uncertain MRI", confidence

    return classes[idx], confidence

# ---------- ROUTE ----------
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
                # Step 2: Real prediction
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
