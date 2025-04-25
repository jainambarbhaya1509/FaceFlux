import os
import cv2
import numpy as np
import insightface
from scipy.spatial import Delaunay
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # Enable CORS if needed

# Initialize InsightFace model
print("🚀 Initializing InsightFace...")
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)

# Global variable to store the reference embedding
reference_embedding = None

@app.route("/load-reference", methods=["POST"])
def load_reference():
    """
    Upload a reference image to compute and store its face embedding.
    """
    global reference_embedding

    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file part in the request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected."}), 400

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    # Read image using OpenCV
    image = cv2.imread(tmp_path)
    os.remove(tmp_path)  # Clean up temporary file

    if image is None:
        return jsonify({"success": False, "message": "Could not read the uploaded image."}), 400

    faces = model.get(image)
    if not faces:
        return jsonify({"success": False, "message": "No face detected in the reference image."}), 400

    reference_embedding = faces[0].embedding
    print("✅ Reference face loaded successfully.")
    return jsonify({"success": True, "message": "Reference face loaded successfully."})

@app.route("/verify", methods=["POST"])
def verify_face():
    """
    Upload a face image for verification against the loaded reference embedding.
    """
    global reference_embedding

    if reference_embedding is None:
        return jsonify({"success": False, "message": "Reference embedding not loaded."}), 400

    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file part in the request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected."}), 400

    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    frame = cv2.imread(tmp_path)
    os.remove(tmp_path)  # Clean up temporary file

    if frame is None:
        return jsonify({"success": False, "message": "Could not read the uploaded image."}), 400

    faces = model.get(frame)
    print(faces)
    if not faces:
        return jsonify({"success": False, "message": "No face detected in the image."}), 400

    threshold = 0.5  # Define a threshold for similarity
    for face in faces:
        similarity = np.dot(reference_embedding, face.embedding) / (
            np.linalg.norm(reference_embedding) * np.linalg.norm(face.embedding)
        )
        if similarity > threshold:
            return jsonify({
                "success": True,
                "verified": True,
                "similarity": float(similarity)
            })

    return jsonify({"success": True, "verified": False})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
