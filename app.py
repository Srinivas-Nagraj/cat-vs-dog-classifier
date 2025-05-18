from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import pickle
from tensorflow import keras
app = Flask(__name__)

# Configure Upload Folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif"}

with open('cnn_model1.pkl', 'rb') as file:
    model = pickle.load(file)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


@app.route("/")
def upload_form():
    return render_template("upload.html")  # Ensure this HTML file is in a 'templates' folder


@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

    image_path = file_path
    image = cv2.imread(image_path)

    test_img = cv2.resize(image, (256, 256))
    test_input = test_img.reshape((1, 256, 256, 3))

    cat = model.predict(test_input)[0][0]
    dog = model.predict(test_input)[0][1]

    if dog >= cat:
        dogpercent=dog*100
        return jsonify({"Predicted animal": "Dog"},{"Percent":float(dogpercent)}), 400
    else:
        catpercent = cat * 100
        return jsonify({"Predicted animal": "Cat"},{"Percent":float(catpercent)}), 400




if __name__ == "__main__":
    app.run(debug=True)
