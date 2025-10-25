from flask import Flask, render_template, request, redirect, url_for, flash
from utils.feature_extractor import extract_features_row
import joblib
import numpy as np
import os
import shutil
import pandas as pd
from werkzeug.utils import secure_filename

## Flask configuration
app = Flask(__name__)
app.secret_key = "urbanecho_secret_key"  # required for flash messages
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["ALLOWED_EXTENSIONS"] = {"wav"}

# Loading the pickle files
MODEL_PATH = "models/pipe.pkl"
FEATURES_PATH = "models/feature_names.pkl"

try:
    pipe = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print("✅ Model and feature names loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    raise SystemExit("Cannot start app without valid model files.")

# Reverse label mapping
LABELS = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music"
}

## Utils
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def clean_uploads_folder():
    """Ensure upload folder exists and is clean before new upload."""
    folder = app.config["UPLOAD_FOLDER"]
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Delete old files
    for f in os.listdir(folder):
        try:
            os.remove(os.path.join(folder, f))
        except Exception:
            pass

# Routes
@app.route("/", methods=["GET"])
def index():
    """Render the upload form page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle uploaded .wav file, extract features, predict class, and render result."""
    if "file" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if file and allowed_file(file.filename):
        clean_uploads_folder()
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        try:
            # Extract features
            features_df = extract_features_row(file_path, feature_names=feature_names)

            # Predict
            probs = pipe.predict_proba(features_df)[0]
            pred_class = np.argmax(probs)
            human_label = LABELS.get(pred_class, "Unknown")

            # Map probabilities to label names
            prob_dict = {LABELS[i]: round(float(p) * 100, 2) for i, p in enumerate(probs)}

            # Clean up file after processing
            os.remove(file_path)

            return render_template(
                "result.html",
                predicted_class=human_label,
                probabilities=prob_dict
            )

        except Exception as e:
            flash(f"Error during processing: {e}")
            try:
                os.remove(file_path)
            except:
                pass
            return redirect(url_for("index"))

    else:
        flash("Invalid file type. Please upload a .wav file.")
        return redirect(url_for("index"))

# Endgame
if __name__ == "__main__":
    app.run(debug=True)
