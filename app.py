
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pymongo import MongoClient
import base64
import uuid
import os
import io  # ✅ Add this line at the top
from PIL import Image
from deepface import DeepFace
import numpy as np
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import zipfile
from datetime import datetime, timedelta
from twilio.rest import Client
import requests
from deepface import DeepFace
import psutil  # 🔍 To log memory usage
import traceback

try:
    print("📦 Preloading DeepFace SFace model...")
    global_model = DeepFace.build_model("SFace")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load SFace model:", str(e))
    traceback.print_exc()
    global_model = None

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ✅ Ensure OPTIONS requests are handled correctly
@app.before_request
def handle_options_request():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 200
# MongoDB Setup
client = MongoClient("mongodb+srv://Aayush:Aayush%402003@photo-gallery.pvd7i.mongodb.net/?retryWrites=true&w=majority&appName=photo-gallery")
db = client["photo_gallery"]
albums_collection = db["albums"]
db = client["dist_and_depart"]
districts_collection = db["districts"]
departments_collection = db["departments"]
auth_db = client["auth_db"]
users_collection = auth_db["users"]
clients_collection = auth_db["clients"]
download_count_collection = auth_db["download-count"]
visitor_collection = auth_db["visitor_logs"]



MODEL_PATH = os.path.join(os.path.dirname(__file__), "facenet_keras.h5")


# Helper function: Compress Image

def compress_image(image_base64, quality=50):
    """
    Decodes a base64 image, compresses it, and returns the compressed image as base64.
    Converts RGBA to RpythonGB if necessary (JPEG does not support transparency).
    """
    try:
        # ✅ Decode base64 image into bytes
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))  # ✅ Convert bytes to PIL Image

        # ✅ Fix: Convert RGBA to RGB before saving as JPEG
        if image.mode == "RGBA":
            image = image.convert("RGB")

        
        output_io = io.BytesIO()
        image.save(output_io, format="JPEG", quality=quality,  optimize=True)
        output_io.seek(0)
        compressed_base64 = base64.b64encode(output_io.getvalue()).decode("utf-8")
        return compressed_base64
    except Exception as e:
        print("Error compressing image:", str(e))
        return None

# Helper function: Extract Face Embeddings
def extract_faces(image_data):
    image_path = f"temp_{uuid.uuid4().hex}.jpg"
    try:
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(image_data))
        
        print(f"\n🔍 Extracting faces from: {image_path}")

        memory = psutil.virtual_memory()
        print(f"💾 Memory - Used: {memory.used // 1024**2}MB, Free: {memory.available // 1024**2}MB, Total: {memory.total // 1024**2}MB")

        if global_model is None:
            print("❌ DeepFace model not loaded.")
            return []

        faces = DeepFace.represent(
            img_path=image_path,
            model_name="SFace",
            # model=global_model,
            enforce_detection=False
        )

        if not faces:
            print("⚠️ No face detected in image.")
            return []

        print(f"✅ Found {len(faces)} face(s)")
        return [
            {
                "face_id": str(uuid.uuid4()),
                "embedding": np.array(face["embedding"]).tolist()
            } for face in faces
        ]

    except Exception as e:
        print("❌ Face extraction failed:", str(e))
        traceback.print_exc()
        return []
    
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)



@app.route("/")
def home():
    return jsonify({"message": "Backend is running successfully!"}), 200
# API: Create Album
@app.route("/create-album", methods=["POST"])
def create_album():
    data = request.json
    album = {
        "_id": str(uuid.uuid4()),
        "name": data["name"],
        "date": data["date"],
        "cover": data["cover"],
        "department": data.get("department", ""),  # ✅ Store department
        "districts": data.get("districts", []),
        "photos": []
    }
    albums_collection.insert_one(album)
    return jsonify({"message": "Album created successfully"}), 201

# API: Upload Photos to Album
@app.route("/upload-gallery/<album_id>", methods=["POST"])
def upload_gallery(album_id):
    data = request.json
    album = albums_collection.find_one({"_id": album_id})
    if not album:
        return jsonify({"error": "Album not found"}), 404

    new_photos = []
    for image in data.get("images", []):  # ✅ Use `.get()` to avoid KeyErrors
        if not image or not isinstance(image, str):
            print("❌ Invalid image received:", image)  # Debugging log
            continue  # Skip invalid images

        compressed_image = compress_image(image)
        if not compressed_image:
            return jsonify({"error": "Image compression failed"}), 500

        new_photos.append({
            "photo_id": str(uuid.uuid4()),
            "image": compressed_image,
            "face_embeddings": extract_faces(compressed_image)
        })

    if new_photos:
        albums_collection.update_one({"_id": album_id}, {"$push": {"photos": {"$each": new_photos}}})
        return jsonify({"message": "Photos uploaded successfully"}), 200
    else:
        return jsonify({"error": "No valid images uploaded"}), 400

# API: Get Albums
@app.route("/albums", methods=["GET"])
def get_albums():
    albums = list(albums_collection.find({}, {"_id": 1, "name": 1, "date": 1, "cover": 1}))
    return jsonify(albums)

# API: Get Photos from Album
@app.route('/photos/<album_id>', methods=['GET'])
def get_album_photos(album_id):
    album = albums_collection.find_one({"_id": album_id})  # ✅ Corrected reference
    if not album:
        return jsonify({"error": "Album not found"}), 404

    photos = album.get("photos", [])
    
    photos_base64 = []
    for photo in photos:
        image_data = photo.get("image")
        if not image_data:
            print(f"Skipping invalid photo: {photo}")
            continue

        if not image_data.startswith("data:image/"):
            image_data = f"data:image/jpeg;base64,{image_data}"
        
        photos_base64.append({
            "photo_id": photo.get("photo_id"),
            "image": image_data
        })

    return jsonify(photos_base64)



# API: Delete Album
@app.route("/album/<album_id>", methods=["DELETE"])
def delete_album(album_id):
    result = albums_collection.delete_one({"_id": album_id})
    if result.deleted_count == 0:
        return jsonify({"error": "Album not found"}), 404
    return jsonify({"message": "Album deleted successfully"}), 200

# API: Delete Photo from Album
@app.route("/photo/<album_id>/<photo_id>", methods=["DELETE"])
def delete_photo(album_id, photo_id):
    print(f"🔍 Deleting photo {photo_id} from album {album_id}")
    
    album = albums_collection.find_one({"_id": album_id}, {"photos": 1})
    if not album:
        return jsonify({"error": "Album not found"}), 404

    print(f"✅ Found album: {album}")

    result = albums_collection.update_one(
    {"_id": album_id},
    {"$pull": {"photos": {"photo_id": str(photo_id)}}}  # ✅ Convert `photo_id` to string
)


    print(f"🛠️ MongoDB update result: {result.raw_result}")

    if result.modified_count == 0:
        return jsonify({"error": "Photo not found"}), 404

    return jsonify({"message": "Photo deleted successfully"}), 200



from bson.objectid import ObjectId  # ✅ Add this import

@app.route("/delete-albums", methods=["DELETE"])
def delete_multiple_albums():
    data = request.json
    album_ids = data.get("albumIds", [])

    if not album_ids:
        return jsonify({"error": "No album IDs provided"}), 400

    result = albums_collection.delete_many({"_id": {"$in": album_ids}})  # Keep IDs as strings

    return jsonify({"message": f"Deleted {result.deleted_count} albums successfully"}), 200



# API: Get all districts
@app.route("/districts", methods=["GET"])
def get_districts():
    districts = list(districts_collection.find({}, {"_id": 0}))
    return jsonify(districts)

# API: Add a new district
@app.route("/districts", methods=["POST"])
def add_district():
    data = request.json
    if "name" not in data:
        return jsonify({"error": "District name is required"}), 400
    districts_collection.insert_one({"name": data["name"]})
    return jsonify({"message": "District added successfully"}), 201

# API: Edit a district
@app.route("/districts/<string:old_name>", methods=["PUT"])
def edit_district(old_name):
    data = request.json
    if "name" not in data:
        return jsonify({"error": "New district name is required"}), 400
    districts_collection.update_one({"name": old_name}, {"$set": {"name": data["name"]}})
    return jsonify({"message": "District updated successfully"}), 200

# API: Delete a district
@app.route("/districts/<string:name>", methods=["DELETE"])
def delete_district(name):
    districts_collection.delete_one({"name": name})
    return jsonify({"message": "District deleted successfully"}), 200

# API: Get all departments
@app.route("/departments", methods=["GET"])
def get_departments():
    departments = list(departments_collection.find({}, {"_id": 0}))
    return jsonify(departments)

# API: Add a new department
@app.route("/departments", methods=["POST"])
def add_department():
    data = request.json
    if "name" not in data:
        return jsonify({"error": "Department name is required"}), 400
    departments_collection.insert_one({"name": data["name"]})
    return jsonify({"message": "Department added successfully"}), 201

# API: Edit a department
@app.route("/departments/<string:old_name>", methods=["PUT"])
def edit_department(old_name):
    data = request.json
    if "name" not in data:
        return jsonify({"error": "New department name is required"}), 400
    departments_collection.update_one({"name": old_name}, {"$set": {"name": data["name"]}})
    return jsonify({"message": "Department updated successfully"}), 200

# API: Delete a department
@app.route("/departments/<string:name>", methods=["DELETE"])
def delete_department(name):
    departments_collection.delete_one({"name": name})
    return jsonify({"message": "Department deleted successfully"}), 200



# API: Add New Staff Member
@app.route("/add-staff", methods=["POST"])
def add_staff():
    data = request.json

    # Validate required fields
    required_fields = ["name", "email", "mobile", "password", "district"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    # Hash the password
    hashed_password = generate_password_hash(data["password"])

    # Create user document
    new_user = {
        "_id": str(uuid.uuid4()),
        "name": data["name"],
        "email": data["email"],
        "mobile": data["mobile"],
        "district": data["district"],
        "role": "Admin",  # Default role
        "password": hashed_password,  # Store only the hashed password
        "status": True  # Default status
    }

    # Insert into database
    users_collection.insert_one(new_user)

    return jsonify({"message": "Staff added successfully"}), 201


# ✅ API to Update User Data
@app.route("/update-user/<string:user_id>", methods=["PUT"])
def update_user(user_id):
    data = request.json
    update_fields = {}

    # Allow status updates along with other fields
    for field in ["name", "email", "mobile", "district", "status"]:
        if field in data:
            update_fields[field] = data[field]

    if not update_fields:
        return jsonify({"error": "No fields to update"}), 400

    result = users_collection.update_one({"_id": user_id}, {"$set": update_fields})

    if result.modified_count == 0:
        return jsonify({"error": "User not found or no changes made"}), 404

    return jsonify({"message": "User updated successfully"}), 200
# API: Get All Users
@app.route("/users", methods=["GET"])
def get_users():
    users = list(users_collection.find({}, {"password": 0})) 
    for user in users:
        user["_id"] = str(user["_id"])  # Convert ObjectId to string
        user.setdefault("photo", "/default-profile.png") # Exclude password
    return jsonify(users)






@app.route("/upload-photo", methods=["POST"])
def upload_photo():
    user_id = request.form.get("userId")
    photo = request.files.get("photo")

    if not user_id or not photo:
        return jsonify({"error": "Missing user ID or photo"}), 400

    # Convert photo to base64
    photo_base64 = base64.b64encode(photo.read()).decode('utf-8')

    # Update the user's profile with base64 photo
    users_collection.update_one({"_id": user_id}, {"$set": {"photo": photo_base64}})

    return jsonify({"message": "Photo uploaded successfully!", "photo": photo_base64}), 200


@app.route("/uploads/<filename>")
def serve_photo(filename):
    return send_file(f"uploads/{filename}", mimetype="image/jpeg")


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    identifier = data.get("identifier")  # Can be username or mobile
    password = data.get("password")

    user = users_collection.find_one({"$or": [{"name": identifier}, {"mobile": identifier}]})
    
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    return jsonify({
        "message": "Login successful",
        "userId": str(user["_id"]),
        "name": user["name"],
        "mobile": user["mobile"],
        "district": user["district"]
    }), 200




@app.route("/get-events", methods=["GET"])
def get_events():
    events = albums_collection.find({}, {"name": 1, "_id": 0})  # Use the correct collection & field name
    event_names = [event["name"] for event in events]  # Extract event names
    return jsonify(event_names)

@app.route("/fetch-album-photos", methods=["POST", "OPTIONS"])  # ✅ Added OPTIONS method
def fetch_album_photos():
    if request.method == "OPTIONS":  
        return jsonify({"message": "CORS Preflight OK"}), 200  # ✅ Handle preflight request

    data = request.json
    event_name = data.get("eventName")

    if not event_name:
        return jsonify({"error": "Event name is required"}), 400

    album = albums_collection.find_one({"name": event_name})

    if not album:
        return jsonify({"error": "No album found with this name"}), 404

    photos = album.get("photos", [])

    return jsonify({
        "photos": [{"photo_id": photo["photo_id"], "image": photo["image"]} for photo in photos]
    })



@app.route("/fetch-photos-by-date", methods=["POST", "OPTIONS"])  # ✅ Handle CORS preflight
def fetch_photos_by_date():
    if request.method == "OPTIONS":  
        return jsonify({"message": "CORS Preflight OK"}), 200  # ✅ CORS preflight response

    data = request.json
    selected_date = data.get("date")

    # Validate input
    if not selected_date:
        return jsonify({"error": "Date is required in YYYY-MM-DD format"}), 400

    # Query all documents where "date" matches the selected_date
    albums = albums_collection.find({"date": selected_date})

    all_photos = []
    
    # Iterate through matching albums and collect all photos
    for album in albums:
        all_photos.extend([
            {"photo_id": photo["photo_id"], "image": photo["image"]}
            for photo in album.get("photos", [])
        ])

    if not all_photos:
        return jsonify({"error": "No photos found for this date"}), 404

    return jsonify({"photos": all_photos})  # ✅ Return all found photos


@app.route('/photo-base64/<photo_id>', methods=['GET'])
def get_photo_base64(photo_id):
    album = albums_collection.find_one({"photos.photo_id": photo_id}, {"photos.$": 1})
    if not album or "photos" not in album or not album["photos"]:
        return jsonify({"error": "Photo not found"}), 404

    photo = album["photos"][0]
    return jsonify({"photo_id": photo["photo_id"], "image": photo["image"]})


 #✅ Updated Route: Search by Uploaded Photo
@app.route("/search-by-upload", methods=["POST"])
def search_by_upload():
    data = request.json
    image_data = data.get("image")
    if not image_data:
        return jsonify({"error": "No image provided"}), 400
    try:
        compressed_data = compress_image(image_data)
        if not compressed_data:
            return jsonify({"error": "Image compression failed"}), 500

        query_embeddings = extract_faces(compressed_data)
        if not query_embeddings:
            return jsonify({"error": "No face found in uploaded photo"}), 404

        matched_photos = []
        all_albums = albums_collection.find()

        for album in all_albums:
            for photo in album.get("photos", []):
                for face in photo.get("face_embeddings", []):
                    for query_face in query_embeddings:
                        emb1 = np.array(face.get("embedding"))
                        emb2 = np.array(query_face.get("embedding"))
                        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                        print(f"🔗 Similarity: {cosine_sim:.4f}")
                        if cosine_sim > 0.7:
                            matched_photos.append({
                                "photo_id": photo.get("photo_id"),
                                "image": photo.get("image")
                            })
                            break

        seen_ids = set()
        unique_photos = []
        for photo in matched_photos:
            if photo["photo_id"] not in seen_ids:
                seen_ids.add(photo["photo_id"])
                unique_photos.append(photo)

        if not unique_photos:
            return jsonify({"error": "No matching faces found in database, either there is no photo of uploaded face or face is not clear "}), 404

        return jsonify({"photos": unique_photos}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/count-albums", methods=["GET"])
def count_albums():
    count = albums_collection.count_documents({})
    return jsonify({"total_albums": count}), 200


@app.route("/count-photos", methods=["GET"])
def count_photos():
    total_photos = 0
    for album in albums_collection.find({}, {"photos": 1}):
        total_photos += len(album.get("photos", []))
    return jsonify({"total_photos": total_photos}), 200

@app.route("/count-users", methods=["GET"])
def count_users():
    try:
        count = users_collection.count_documents({})
        return jsonify({"total_users": count}), 200
    except Exception as e:
        print("❌ Error in /count-users:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/increment-download-count", methods=["POST"])
def increment_download_count():
    try:
        # You can track count per album or just total
        result = download_count_collection.find_one({"_id": "total"})
        if result:
            download_count_collection.update_one({"_id": "total"}, {"$inc": {"count": 1}})
        else:
            download_count_collection.insert_one({"_id": "total", "count": 1})
        return jsonify({"message": "Download count updated"}), 200
    except Exception as e:
        print("❌ Error incrementing download count:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/get-download-count", methods=["GET"])
def get_download_count():
    count_doc = auth_db["download-count"].find_one({"_id": "total"})
    return jsonify({"count": count_doc["count"] if count_doc else 0}), 200

@app.route("/record-visit", methods=["POST"])
def record_visit():
    try:
        visitor_collection.insert_one({
            "timestamp": datetime.utcnow()
        })
        return jsonify({"message": "Visit recorded"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/visitor-stats", methods=["GET"])
def visitor_stats():
    try:
        today = datetime.utcnow().date()
        last_7_days = [today - timedelta(days=i) for i in range(6, -1, -1)]
        
        pipeline = [
            {
                "$group": {
                    "_id": {
                        "year": {"$year": "$timestamp"},
                        "month": {"$month": "$timestamp"},
                        "day": {"$dayOfMonth": "$timestamp"}
                    },
                    "count": {"$sum": 1}
                }
            }
        ]
        raw_data = list(visitor_collection.aggregate(pipeline))

        # Format as {date_string: count}
        count_map = {
            f"{d['_id']['year']}-{d['_id']['month']:02d}-{d['_id']['day']:02d}": d["count"]
            for d in raw_data
        }

        # Prepare 7-day output
        results = []
        for day in last_7_days:
            key = day.strftime("%Y-%m-%d")
            results.append({
                "date": key,
                "count": count_map.get(key, 0)
            })

        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/master-search", methods=["POST"])
def master_search():
    data = request.json
    query = data.get("query", "").strip().lower()

    if not query:
        return jsonify({"error": "Empty search"}), 400

    matching_photos = []

    albums = albums_collection.find()

    for album in albums:
        # Simple case-insensitive match
        album_name = album.get("name", "").lower()
        department = album.get("department", "").lower()
        districts = [d.lower() for d in album.get("districts", [])]

        if (
            query in album_name or
            query in department or
            any(query in d for d in districts)
        ):
            matched_by = []
            if query in album_name: matched_by.append("Event")
            if query in department: matched_by.append("Department")
            if any(query in d for d in districts): matched_by.append("District")

            for photo in album.get("photos", []):
                matching_photos.append({
                    "photo_id": photo["photo_id"],
                    "image": photo["image"],
                    "matched_by": matched_by,
                    "album_name": album.get("name", ""),
                    "department": album.get("department", ""),
                    "districts": album.get("districts", [])
                })

    return jsonify({"photos": matching_photos})

@app.route("/search-suggestions", methods=["GET"])
def search_suggestions():
    events = [e["name"] for e in albums_collection.find({}, {"name": 1})]
    departments = [d["name"] for d in departments_collection.find({}, {"name": 1})]
    districts = [d["name"] for d in districts_collection.find({}, {"name": 1})]

    return jsonify({
        "events": list(set(events)),
        "departments": list(set(departments)),
        "districts": list(set(districts))
    })

@app.route("/fetch-all-photos", methods=["GET"])
def fetch_all_photos():
    all_photos = []
    for album in albums_collection.find({}, {"photos": 1}):
        all_photos.extend(album.get("photos", []))

    result = [
        {
            "photo_id": p.get("photo_id"),
            "image": f"data:image/jpeg;base64,{p['image']}" if not p["image"].startswith("data:image") else p["image"]
        }
        for p in all_photos if "photo_id" in p and "image" in p
    ]

    return jsonify(result), 200



MSG91_AUTHKEY = "189400AF2q5EpqOYU67f60efbP1"
MSG91_SENDER_ID = "CMOPAI"  # e.g., MSGIND
MSG91_TEMPLATE_ID = "61161dfa1848146fb7608293"  # from your approved MSG91 templates

otp_store = {}

@app.route("/send-otp", methods=["POST"])
def send_otp():
    data = request.json
    mobile = data.get("mobile")

    if not mobile:
        return jsonify({"error": "Mobile number is required"}), 400

    otp = str(uuid.uuid4().int)[:6]
    otp_store[mobile] = otp

    # MSG91 SMS API Payload
    url = "https://api.msg91.com/api/v5/otp"
    payload = {
        "template_id": MSG91_TEMPLATE_ID,
        "mobile": f"91{mobile}",  # Indian mobile format
        "authkey": MSG91_AUTHKEY,
        "otp": otp,
        "sender": MSG91_SENDER_ID
    }

    try:
        res = requests.post(url, json=payload)
        result = res.json()
        if res.status_code == 200:
            return jsonify({"message": "OTP sent successfully!"}), 200
        else:
            return jsonify({"error": result.get("message", "Failed to send OTP")}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/verify-otp", methods=["POST"])
def verify_otp():
    data = request.json
    mobile = data.get("mobile")
    otp = data.get("otp")

    if not mobile or not otp:
        return jsonify({"error": "Mobile and OTP are required"}), 400

    if otp_store.get(mobile) != otp:
        return jsonify({"error": "Invalid OTP"}), 401

    del otp_store[mobile]  # ✅ Remove OTP after use

    return jsonify({"message": "OTP verified successfully"}), 200

@app.route("/complete-signup", methods=["POST"])
def complete_signup():
    data = request.json
    required = ("name", "email", "password", "district")

    if not all(data.get(k) for k in required):
        return jsonify({"error": "Missing fields"}), 400

    # ✅ Get verified mobile from header or fallback
    verified_mobile = request.headers.get("X-Otpless-Mobile") or data.get("mobile")

    if not verified_mobile:
        return jsonify({"error": "Mobile number missing or not verified"}), 400

    # ✅ Check if already registered
    if clients_collection.find_one({"mobile": verified_mobile}):
        return jsonify({"error": "Mobile already registered"}), 409

    try:
        with open("public/pro.png", "rb") as f:
            photo_base64 = base64.b64encode(f.read()).decode("utf-8")
    except:
        photo_base64 = ""

    new_user = {
        "_id": str(uuid.uuid4()),
        "name": data["name"],
        "email": data["email"],
        "mobile": verified_mobile,
        "district": data["district"],
        "role": "User",
        "status": True,
        "photo": photo_base64,
        "password": generate_password_hash(data["password"]),
    }

    clients_collection.insert_one(new_user)

    return jsonify({
        "message": "User registered successfully",
        "userId": new_user["_id"],
        "name": new_user["name"],
        "mobile": new_user["mobile"],
        "district": new_user["district"]
    }), 200


@app.route("/client-login", methods=["POST"])
def client_login():
    data = request.json
    mobile = data.get("mobile")
    password = data.get("password")

    if not mobile or not password:
        return jsonify({"error": "Mobile and password are required"}), 400

    # Look up client by mobile
    client = clients_collection.find_one({"mobile": mobile})
    
    if not client or not check_password_hash(client["password"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    return jsonify({
        "message": "Login successful",
        "userId": str(client["_id"]),
        "name": client["name"],
        "mobile": client["mobile"],
        "district": client.get("district", ""),
        "role": client.get("role", "User")
    }), 200

@app.route("/update-client/<string:user_id>", methods=["PUT"])
def update_client(user_id):
    data = request.json
    allowed_fields = ["name", "mobile", "district"]

    update_fields = {field: data[field] for field in allowed_fields if field in data}

    if not update_fields:
        return jsonify({"error": "No valid fields to update"}), 400

    result = clients_collection.update_one({"_id": user_id}, {"$set": update_fields})

    if result.modified_count == 0:
        return jsonify({"error": "Client not found or no changes made"}), 404

    return jsonify({"message": "Client updated successfully"}), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))  # Get PORT from Render, default to 5000
    app.run(host="0.0.0.0", port=port, debug=False)




