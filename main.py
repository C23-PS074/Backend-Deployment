from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import storage
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import bcrypt
import mysql.connector
from datetime import datetime, timedelta

# from users import users_bp
app = Flask(__name__)
CORS(app)

db = mysql.connector.connect(
    host='10.119.224.3',
    user='root',
    password='fracturevision',
    database='fracturevisiondb'
)

cursor = db.cursor()


# Menginisialisasi penyimpanan Google Cloud
storage_client = storage.Client()
bucket_name = 'fracturevisionbucket'
model_file_name = 'fracture_classification_model.h5'

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

def upload_image_to_bucket(image):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_filename = f"{timestamp}_{image.filename}"
    folder_name = "uploads"
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(folder_name + "/" + unique_filename)
    blob.upload_from_file(image)
    image_path = f"https://storage.googleapis.com/{bucket_name}/{folder_name}/{unique_filename}"

    return image_path
    print(bucket_name)
    print("Hubla")

# Handle the image upload and prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        image = request.files["image"]
        users_id = request.form.get('id')
        uploaded_image_path = upload_image_to_bucket(image)
        img = Image.open(image)
        img = img.convert("RGB")
        img = preprocess_image(img)

        # Make the prediction
        prediction = model.predict(img)

        if prediction[0] < 0.5:
            predicted_class = 'Fractured'
        else:
            predicted_class = 'Normal'

        dateNow = datetime.date(datetime.now())
        timeNow = datetime.now() + timedelta(hours=7)

        query = "INSERT INTO record (image, result, date, time, users_id) VALUES (%s, %s, %s, %s, %s)"
        values = (uploaded_image_path, predicted_class, dateNow, timeNow, users_id)
        cursor.execute(query, values)
        db.commit()

        return {"prediction": predicted_class, "image_path": uploaded_image_path}
    except Exception as e:
        return {"error": str(e)}





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
            "error": true,
            "message": "Record Empty",
            "datarecord": []
        }
        return jsonify(response_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

# Menutup koneksi ke database
cursor.close()
db.close()