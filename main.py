from flask import Flask, request
from flask_cors import CORS
from google.cloud import storage
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Menginisialisasi penyimpanan Google Cloud
storage_client = storage.Client()
bucket_name = 'fracturevisionbucket'
model_file_name = 'fractured_model.h5'

# Mendownload file model dari bucket
bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(model_file_name)
model_path = '/tmp/fractured_model.h5'
blob.download_to_filename(model_path)

# Memuat model
model = tf.keras.models.load_model(model_path)

# Preprocess the image
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Handle the image upload and prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        image = request.files["image"]
        img = Image.open(image)
        img = img.convert("RGB")
        img = preprocess_image(img)

        # Make the prediction
        prediction = model.predict(img)

        if prediction[0] < 0.5:
            predicted_class = 'Fractured'
        else:
            predicted_class = 'Normal'
        print(prediction[0])
        print(predicted_class)
        return {"prediction": predicted_class}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
