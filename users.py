from flask import Blueprint, request
import bcrypt
import mysql.connector

users_bp = Blueprint('users_bp', __name__)

# Membuat koneksi ke database
db = mysql.connector.connect(
    host: '10.119.224.3',
    user: 'root',
    password: 'fracturevision',
    database: 'fracturevisiondb'
  
)

# Membuat cursor untuk eksekusi perintah SQL
cursor = db.cursor()

@users_bp.route('/register', methods=['POST'])
def register(username, password, fullname, phone, address, role):
    # Mengecek apakah username sudah terdaftar
    query = "SELECT * FROM users WHERE username = %s"
    cursor.execute(query, (username,))
    if cursor.fetchone():
        return "Username sudah terdaftar"

    # Mengenkripsi password
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    # Menyimpan data pengguna ke database
    query = "INSERT INTO users (username, password, fullname, phone, address, role) VALUES (%s, %s, %s, %s, %s, %s)"
    values = (username, hashed_password, fullname, phone, address, role)
    cursor.execute(query, values)
    db.commit()

    return "Registrasi berhasil"


@users_bp.route('/login', methods=['POST'])
def login(username, password):
    # Mengecek apakah username ada dalam database
    query = "SELECT * FROM users WHERE username = %s"
    cursor.execute(query, (username,))
    user = cursor.fetchone()
    if not user:
        return "Username tidak terdaftar"

    # Memeriksa kesesuaian password
    hashed_password = user[2]
    if bcrypt.checkpw(password.encode(), hashed_password.encode()):
        return "Login berhasil"
    else:
        return "Password salah"

# Registrasi pengguna
result = register(username, password, fullname, phone, address, role)
print(result)

# Login pengguna
login_result = login(username, password)
print(login_result)

# Menutup koneksi ke database
cursor.close()
db.close()
