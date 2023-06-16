from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import storage
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import io
from io import BytesIO
import bcrypt
import mysql.connector
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
import scipy.misc
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import ops as utils_ops
import argparse
import glob
import os
import zipfile

# from users import users_bp
app = Flask(__name__)
CORS(app)

db = mysql.connector.connect(
    host='10.119.224.3',
    user='root',
    password='fracturevision',
    database='fracturevisiondb'
)

cursor = db.cursor(buffered=True)

# Menginisialisasi penyimpanan Google Cloud
storage_client = storage.Client()
bucket_name = 'fracturevisionbucket'
model_file_name = 'fracture_classification_model.h5'
file_zip = 'fracture_detection_model.zip'

# Mendownload file model dari bucket
bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(model_file_name)
predict_model_path = '/tmp/fractured_model.h5'
blob.download_to_filename(predict_model_path)

blob = bucket.blob(model_file_name)
zip_file_path = '/tmp/fracture_detection_model.zip'
blob.download_to_filename(zip_file_path)

model_path = f"https://storage.googleapis.com/fracturevisionbucket/inference_graph/saved_model/"
label_map_path = f"https://storage.googleapis.com/fracturevisionbucket/bone-fractures_label_map.pbtxt"

# Memuat model
predict_model = tf.keras.models.load_model(predict_model_path)

utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile

# Preprocess the image
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def upload_image_to_bucket(image):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_filename = f"{timestamp}_{image.filename}"
    folder_name = "uploads"
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(folder_name + "/" + unique_filename)
    blob.upload_from_file(image)
    image_path = f"https://storage.googleapis.com/{bucket_name}/{folder_name}/{unique_filename}"

    return image_path


#DETEKSI DARI SINI

def load_model_detection(model_path):
    model = tf.saved_model.load(model_path)
    return model

detection_model = load_model_detection(model_path)

def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def detection_output(model, image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    output_dict = model(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'fracture' in output_dict:
        detection_fracture_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['fracture'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_fracture_reframed = tf.cast(detection_fracture_reframed > 0.5, tf.uint8)
        output_dict['detection_fracture_reframed'] = detection_fracture_reframed.numpy()

    return output_dict


def run_detection(model, category_index, uploaded_image_path,output_path, fileimage, timestamp):
    image_np = load_image_into_numpy_array(uploaded_image_path)
    output_dict = detection_output(model, image_np)
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        skip_labels=True,
        max_boxes_to_draw=200,
        line_thickness=8)
    plt.imshow(image_np)
    waktu=timestamp
    uniquename= fileimage
    image2 = Image.fromarray(image_np)
    image2.save('/tmp/hasil_'+ waktu + '_' + uniquename)

    file_path1 = f"/tmp/hasil_{uniquename}"
    if os.path.exists(file_path1):
        print("File exists.")
    else:
        print("File does not exist.")

    for detection in output_dict['detection_scores']:
        if (detection >= 0.5):
            return True
        else:
            return False


@app.route("/predict", methods=["POST"])
def detection():
    try:
        output_path = '/tmp/output'
        image = request.files["image"]
        users_id = request.form.get('id')
        fileimage = image.filename
        uploaded_image_path = upload_image_to_bucket(image)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        img = Image.open(image)
        img = img.convert("RGB")
        img = preprocess_image(img)

        # Make the prediction
        prediction = predict_model.predict(img)
        
        if prediction[0] < 0.5:
            predicted_class = 'Fractured'
            category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=False)
            is_detected = run_detection(detection_model, category_index, uploaded_image_path,output_path, fileimage, timestamp)
                
            bucket = storage_client.get_bucket(bucket_name)
            unique_file_name = f"hasil_{timestamp}_{image.filename}"
            foldername="output"
            blob = bucket.blob(foldername + "/" + unique_file_name)
            blob.upload_from_filename("/tmp/"+ unique_file_name)
            detectionimage_path= f"https://storage.googleapis.com/{bucket_name}/{foldername}/{unique_file_name}"
            path_response = detectionimage_path;
        else:
            predicted_class = 'Normal'
            path_response = uploaded_image_path;


        wibtime = datetime.utcnow() + timedelta(hours=7)

        dateNow = datetime.date(datetime.now())
        timeNow = wibtime.time()

        query = "INSERT INTO record (image, result, date, time, users_id) VALUES (%s, %s, %s, %s, %s)"
        values = (path_response, predicted_class, dateNow, timeNow, users_id)
        cursor.execute(query, values)
        db.commit()

        detection={"prediction": predicted_class, 
                "image_path": path_response}
        
        # detection={"prediction": predicted_class, 
        #         "image_path": uploaded_image_path,
        #         "detection_path": detectionimage_path}
        return detection
    except Exception as e:
        return {"error": str(e)}



# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         image = request.files["image"]
#         users_id = request.form.get('id')
#         uploaded_image_path = upload_image_to_bucket(image)
#         img = Image.open(image)
#         img = img.convert("RGB")
#         img = preprocess_image(img)

#         # Make the prediction
#         prediction = predict_model.predict(img)

#         if prediction[0] < 0.5:
#             predicted_class = 'Fractured'
#         else:
#             predicted_class = 'Normal'


#         wibtime = datetime.utcnow() + timedelta(hours=7)

#         dateNow = datetime.date(datetime.now())
#         timeNow = wibtime.time()

#         query = "INSERT INTO record (image, result, date, time, users_id) VALUES (%s, %s, %s, %s, %s)"
#         values = (uploaded_image_path, predicted_class, dateNow, timeNow, users_id)
#         cursor.execute(query, values)
#         db.commit()

#         return {"prediction": predicted_class, "image_path": uploaded_image_path}
#     except Exception as e:
#         return {"error": str(e)}





@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username')
    password = request.form.get('password')
    fullname = request.form.get('fullname')
    phone = request.form.get('phone')
    address = request.form.get('address')

    if username=='':
        respond = {
        'error': True,
        'message': 'username harus diisi',
    }
        return respond
    else:
        if password=='':
            respond = {
            'error': True,
            'message': 'password harus diisi',
        }
            return respond
        else:
            if fullname=='':
                respond = {
                'error': True,
                'message': 'nama lengkap harus diisi',
            }
                return respond
            else:
                if phone=='':
                    respond = {
                    'error': True,
                    'message': 'nomor telepon harus diisi',
            }
                    return respond
                else:
                    if address=='':
                        respond = {
                        'error': True,
                        'message': 'alamat harus diisi',
                }
                        return respond


    # Mengecek apakah username sudah terdaftar
    query = "SELECT * FROM users WHERE username = %s"
    cursor.execute(query, (username,))
    if cursor.fetchone():
        respond = {
            'error': True,
            'message': 'Username sudah terdaftar',
        }
        return respond

    # Mengenkripsi password
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    # Menyimpan data pengguna ke database
    query = "INSERT INTO users (username, password, fullname, phone, address) VALUES (%s, %s, %s, %s, %s)"
    values = (username, hashed_password, fullname, phone, address)
    cursor.execute(query, values)
    db.commit()

    respond = {
        'error': False,
        'message' : 'Registrasi Berhasil',
    }
    return respond


@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    # Mengecek apakah username ada dalam database
    query = "SELECT * FROM users WHERE username = %s"
    cursor.execute(query, (username,))
    user = cursor.fetchone()
    if not user:
        respond = {
            'error': True,
            'message': 'Username tidak terdaftar'
        }
        return respond

    # Memeriksa kesesuaian password
    hashed_password = user[2]
    if bcrypt.checkpw(password.encode(), hashed_password.encode()):
        user_data = {
            "id": user[0],
            "username": user[1],
            "fullname": user[3],
            "phone": user[4],
            "address": user[5]
        }

        return user_data
    else:
        respond = {
            'error': True,
            'message': 'Password salah'
        }
        return respond


@app.route("/users", methods=["GET"])
def users():
    try:
        select_query = "SELECT * FROM users"
        cursor.execute(select_query)
        results = cursor.fetchall()
        print(results)
        print("HUBLA")
        return results
    except Exception as e:
        return {"error": str(e)}


@app.route("/record", methods=["GET"])
def record():
    try:
        users_id = request.args.get('id')
        select_query = "SELECT * FROM record WHERE users_id = %s"
        cursor.execute(select_query, (users_id,))
        results = cursor.fetchall()

        # Mengonversi hasil kueri menjadi format yang dapat di-serialisasi
        formatted_results = []
        for row in results:
            formatted_row = {
                "id": row[0],
                "image": row[1],
                "result": row[2],
                "date": str(row[3]),  # Mengonversi tanggal menjadi string
                "time": str(row[4])  # Mengonversi waktu menjadi string
            }
            formatted_results.append(formatted_row)
            response_data = {
                "error": False,
                "message": "Record fetched successfully",
                "datarecord": formatted_results
            }
        return jsonify(response_data)
    except Exception as e:
        response_data = {
            "error": True,
            "message": "Record Empty",
            "datarecord": []
        }
        return jsonify(response_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

# Menutup koneksi ke database
cursor.close()
db.close()