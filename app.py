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

# MRI validation
def is_mri_image(path):
    img = cv2.imread(path, 0)
    if img is None:
        return False
    variance = np.var(img)
    return 500 < variance < 5000

# Real AI prediction
def predict_tumor(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=(0,1))

    inputs = {session.get_inputs()[0].name: img}
    outputs = session.run(None, inputs)[0][0]

    idx = np.argmax(outputs)
    confidence = round(float(outputs[idx]) * 100, 2)

    return classes[idx], confidence

@app.route("/", methods=["GET", "POST"])
def index():
    result = confidence = image_path = message = None

    if request.method == "POST":
        file = request.files["file"]
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
