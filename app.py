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
from deepface.DeepFace import build_model
facenet_model = build_model("Facenet")
import pymongo
import certifi
import boto3
from botocore.client import Config
from bson.objectid import ObjectId 
from flask import send_file
import requests
from datetime import datetime 
from dotenv import load_dotenv
load_dotenv()
from urllib.parse import urlparse



# Cloudflare R2 credentials
# R2_ACCOUNT_ID = "7ba78c8bca1993356ed4787cee42d111"
# R2_ACCESS_KEY_ID = "cb70427b13bece34cef2f9bca5b08b6a"
# R2_SECRET_ACCESS_KEY = "965be11d5a0247c43d4510bbc9b3cebe7da55406c7ba5d49b1967698960fe4c6"
# R2_BUCKET_NAME = "photo-gallery-bucket"
# R2_REGION = "auto"  # Usually 'auto' for Cloudflare
# PUBLIC_BUCKET_DOMAIN = "pub-b067d59ae9cd4e1797621c719e4f31e3.r2.dev"


R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_REGION = os.getenv("R2_REGION")
PUBLIC_BUCKET_DOMAIN = os.getenv("PUBLIC_BUCKET_DOMAIN")


# ✅ Set your R2 endpoint URL
R2_ENDPOINT_URL = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

s3_client = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    config=Config(signature_version="s3v4"),
    region_name=R2_REGION,
)


print("🔧 Loading Facenet model...")
facenet_model = DeepFace.build_model("Facenet")
print("✅ Facenet model loaded once.")



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ✅ Ensure OPTIONS requests are handled correctly
@app.before_request
def handle_options_request():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 200
    

client = pymongo.MongoClient(
    "mongodb+srv://Aayush:Aayush%402003@photo-gallery.pvd7i.mongodb.net/?retryWrites=true&w=majority&appName=photo-gallery",
    tls=True,
    tlsCAFile=certifi.where()  # Add this line
 )


# client = MongoClient("mongodb://localhost:27017/")

photo_gallery_db = client["photo_gallery"]
albums_collection = photo_gallery_db["albums"]

dist_and_depart_db = client["dist_and_depart"]
districts_collection = dist_and_depart_db["districts"]
departments_collection = dist_and_depart_db["departments"]
banners_collection = dist_and_depart_db["banners"]


auth_db = client["auth_db"]
users_collection = auth_db["users"]
clients_collection = auth_db["clients"]
download_count_collection = auth_db["download-count"]
visitor_collection = auth_db["visitor_logs"]


MODEL_PATH = os.path.join(os.path.dirname(__file__), "facenet_keras.h5")



def upload_to_r2(base64_image, filename):
    try:
        image_data = base64.b64decode(base64_image)
        s3_client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=filename,
            Body=image_data,
            ContentType="image/jpeg",
            ACL='public-read'
        )
        public_url = f"https://{PUBLIC_BUCKET_DOMAIN}/{filename}"

        return public_url
    except Exception as e:
        print("❌ Upload to R2 failed:", str(e))
        return None
    

def delete_from_r2(file_url):
    try:
        # Extract the filename from URL
        if not file_url:
            return
        key = file_url.split(PUBLIC_BUCKET_DOMAIN + "/")[-1]
        s3_client.delete_object(Bucket=R2_BUCKET_NAME, Key=key)
        print(f"✅ Deleted {key} from R2")
    except Exception as e:
        print(f"❌ Failed to delete {file_url}: {str(e)}")


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
    with open(image_path, "wb") as f:
        f.write(base64.b64decode(image_data))

    try:
        print(f"🔍 Extracting faces from: {image_path}")
        
        faces = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            
            enforce_detection=False
        )

        os.remove(image_path)
        print(f"✅ Found {len(faces)} face(s)")
        return [
            {
                "face_id": str(uuid.uuid4()),
                "embedding": np.array(face["embedding"]).tolist()
            } for face in faces
        ]

    except Exception as e:
        print("❌ Face extraction failed:", str(e))
        os.remove(image_path)
        return []


@app.route("/events-by-date", methods=["POST"])
def get_events_by_date():
    data = request.json
    selected_date = data.get("date")

    if not selected_date:
        # If no date provided, return latest 10 unique events
        events = albums_collection.find({}, {"name": 1, "date": 1}).sort("date", -1).limit(20)

        seen = set()
        latest_events = []
        for event in events:
            name = event.get("name")
            if name and name not in seen:
                seen.add(name)
                latest_events.append(name)
            if len(latest_events) == 10:
                break

        return jsonify(latest_events), 200

    albums = albums_collection.find({"date": selected_date}, {"name": 1})
    event_names = list(set(album["name"] for album in albums))

    return jsonify(event_names), 200

@app.route("/")
def home():
    return jsonify({"message": "Backend is running successfully!"}), 200


# API: Create Album
@app.route("/create-album", methods=["POST"])
def create_album():
    data = request.json

    # Upload cover image to R2
    compressed_cover = compress_image(data["cover"])
    cover_filename = f"covers/{uuid.uuid4().hex}.jpg"
    cover_url = upload_to_r2(compressed_cover, cover_filename)

    album = {
        "_id": str(uuid.uuid4()),
        "name": data["name"],
        "date": data["date"],
        "cover": cover_url,  # ✅ Save URL instead of base64
        "department": data.get("department", ""),
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
    for image in data.get("images", []):
        if not image or not isinstance(image, str):
            continue  # Skip invalid images

        compressed_image = compress_image(image)
        if not compressed_image:
            continue  # Skip if compression failed

        # Upload photo to R2
        photo_filename = f"photos/{uuid.uuid4().hex}.jpg"
        photo_url = upload_to_r2(compressed_image, photo_filename)

        new_photos.append({
            "photo_id": str(uuid.uuid4()),
            "image": photo_url,  # ✅ Save URL instead of base64
            "face_embeddings": extract_faces(compressed_image)  # Still pass compressed image for face
        })

    if new_photos:
        albums_collection.update_one({"_id": album_id}, {"$push": {"photos": {"$each": new_photos}}})
        return jsonify({"message": "Photos uploaded successfully"}), 200
    else:
        return jsonify({"error": "No valid images uploaded"}), 400

# API: Get Albums
@app.route("/albums", methods=["GET"])
def get_albums():
    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 16))
        skip = (page - 1) * limit

        albums = albums_collection.aggregate([
            {"$limit": limit},
            {"$skip": skip},
            {"$project": {
                "_id": 1,
                "name": 1,
                "date": 1,
                "cover": 1,
                 "districts": 1,
                "photo_count": {"$size": {"$ifNull": ["$photos", []]}}
            }}
        ])
        albums = list(albums)

        total_count = albums_collection.count_documents({})

        return jsonify({
            "albums": albums,
            "total": total_count
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/photos/<album_id>", methods=["GET"])
def get_album_photos(album_id):
    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get('limit', 16))

        album = albums_collection.find_one({"_id": album_id})   # ✅ First fetch album
        if not album:
            return jsonify({"error": "Album not found"}), 404

        photos = album.get("photos", [])
        total_photos = len(photos)

        if limit == 0:
            paginated_photos = photos  # return all
        else:
            skip = (page - 1) * limit
            paginated_photos = photos[skip:skip + limit]


        formatted_photos = [
            {
                "photo_id": p.get("photo_id"),
                "image": p.get("image")  # ✅ Direct R2 URL
            }
            for p in paginated_photos
        ]


        return jsonify({
            "photos": formatted_photos,
            "total": total_photos
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# API: Delete Album
@app.route("/album/<album_id>", methods=["DELETE"])
def delete_album(album_id):
    album = albums_collection.find_one({"_id": album_id})
    if not album:
        return jsonify({"error": "Album not found"}), 404

    # ✅ Delete cover image from R2
    delete_from_r2(album.get("cover"))

    # ✅ Delete all photos from R2
    for photo in album.get("photos", []):
        delete_from_r2(photo.get("image"))

    # ✅ Now delete from MongoDB
    albums_collection.delete_one({"_id": album_id})

    return jsonify({"message": "Album and its photos deleted successfully"}), 200


# API: Delete Photo from Album
@app.route("/photo/<album_id>/<photo_id>", methods=["DELETE"])
def delete_photo(album_id, photo_id):
    album = albums_collection.find_one({"_id": album_id}, {"photos": 1})
    if not album:
        return jsonify({"error": "Album not found"}), 404

    photo_to_delete = next((photo for photo in album.get("photos", []) if photo.get("photo_id") == photo_id), None)

    if not photo_to_delete:
        return jsonify({"error": "Photo not found"}), 404

    # ✅ Delete the image from R2
    delete_from_r2(photo_to_delete.get("image"))

    # ✅ Remove photo from album in MongoDB
    albums_collection.update_one({"_id": album_id}, {"$pull": {"photos": {"photo_id": photo_id}}})

    return jsonify({"message": "Photo deleted successfully"}), 200



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

    for field in ["name", "email", "mobile", "district", "status"]:
        if field in data:
            update_fields[field] = data[field]

    if not update_fields:
        return jsonify({"error": "No fields to update"}), 400

    # Try users collection first
    result = users_collection.update_one({"_id": user_id}, {"$set": update_fields})

    if result.modified_count == 0:
        # Try clients collection if not found in users
        result = clients_collection.update_one({"_id": user_id}, {"$set": update_fields})
        if result.modified_count == 0:
            return jsonify({"error": "User not found or no changes made"}), 404

    return jsonify({"message": "User updated successfully"}), 200



# API: Get All Users
@app.route("/users", methods=["GET"])
def get_users():
    projection = {"name": 1, "email": 1, "mobile": 1, "district": 1, "role": 1, "status": 1}
    
    users = list(users_collection.find({}, projection))
    clients = list(clients_collection.find({}, projection))

    combined_users = users + clients

    for user in combined_users:
        user["_id"] = str(user["_id"])
        user["role"] = user.get("role", "User")
        user["status"] = user.get("status", True)
        user["mobile"] = user.get("mobile") or "Gmail User"
        user["district"] = user.get("district") or "Gmail User"

    return jsonify(combined_users)



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
                        if cosine_sim > 0.5:
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
        user_count = users_collection.count_documents({})
        client_count = clients_collection.count_documents({})
        total = user_count + client_count
        return jsonify({"total_users": total}), 200
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
    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 16))
        skip = (page - 1) * limit

        all_photos = []
        for album in albums_collection.find({}, {"photos": 1}):
            all_photos.extend(album.get("photos", []))

        total = len(all_photos)
        paginated = all_photos[skip:skip + limit]

        result = [
            {
                "photo_id": p.get("photo_id"),
                "image": p.get("image")  # ✅ Direct R2 URL
            }
            for p in paginated if "photo_id" in p and "image" in p
        ]


        return jsonify({"photos": result, "total": total}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/complete-signup", methods=["POST"])
def complete_signup():
    data = request.json
    required = ("name", "email", "password", "district")

    if not all(data.get(k) for k in required):
        return jsonify()

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

    client = clients_collection.find_one({"mobile": mobile})

    if not client or not check_password_hash(client["password"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    if not client.get("status", True):
        return jsonify({"error": "Your account is inactive. Please contact admin."}), 403

    return jsonify({
        "message": "Login successful",
        "userId": str(client["_id"]),
        "name": client["name"],
        "mobile": client["mobile"],
        "email": client.get("email", ""),
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



@app.route("/check-user-exists", methods=["POST"])
def check_user_exists():
    data = request.json
    mobile = data.get("mobile")
    email = data.get("email")

    if not mobile or not email:
        return jsonify({"error": "Mobile and Email required"}), 400

    # Check for mobile
    if clients_collection.find_one({"mobile": mobile}):
        return jsonify({"error": "Mobile number already registered"}), 409

    # Check for email
    if clients_collection.find_one({"email": email}):
        return jsonify({"error": "Email already registered"}), 409

    return jsonify({"message": "Mobile and Email are available"}), 200



@app.route("/reset-password", methods=["POST"])
def reset_password():
    data = request.json
    mobile = data.get("mobile")
    new_password = data.get("newPassword")

    if not mobile or not new_password:
        return jsonify({"error": "Mobile and new password are required"}), 400

    user = clients_collection.find_one({"mobile": mobile})
    if not user:
        return jsonify({"error": "User not found"}), 404

    hashed_password = generate_password_hash(new_password)

    clients_collection.update_one(
        {"mobile": mobile},
        {"$set": {"password": hashed_password}}
    )

    return jsonify({"success": True, "message": "Password updated successfully"}), 200




@app.route("/albums-by-district", methods=["GET"])
def get_albums_by_district():
    district_name = request.args.get("name")
    if not district_name:
        return jsonify({"error": "District name is required"}), 400

    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 16))
        skip = (page - 1) * limit

        query = {"districts": {"$in": [district_name]}}
        total = albums_collection.count_documents(query)

        albums = list(albums_collection.aggregate([
            {"$match": query},
            {"$skip": skip},
            {"$limit": limit},
            {"$project": {
                "_id": 1,
                "name": 1,
                "date": 1,
                "cover": 1,
                "photo_count": {"$size": {"$ifNull": ["$photos", []]}}  # ✅ Add this line
            }}
        ]))

        return jsonify({
            "albums": albums,
            "total": total
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/proxy-image")
def proxy_image():
    image_url = request.args.get("url")
    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    try:
        response = requests.get(image_url, stream=True)
        return send_file(io.BytesIO(response.content), mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@app.route("/google-login", methods=["POST"]) 
def google_login():
    data = request.json
    email = data.get("email")

    if not email:
        return jsonify({"error": "Email is required"}), 400

    existing_user = clients_collection.find_one({"email": email})
    
    if existing_user:
        if not existing_user.get("status", True):
            return jsonify({"error": "Your account is inactive. Please contact admin."}), 403
        return jsonify({
            "message": "User already exists",
            "userId": existing_user["_id"]
        }), 200

    try:
        photo = data.get("photo")
        if not photo:
            with open("public/pro.png", "rb") as f:
                photo = base64.b64encode(f.read()).decode("utf-8")

        new_user = {
            "_id": str(uuid.uuid4()),
            "name": data.get("name"),
            "email": email,
            "mobile": "",
            "district": "",
            "role": "User",
            "status": True,
            "photo": photo,
            "password": ""
        }

        clients_collection.insert_one(new_user)

        return jsonify({
            "message": "Google user registered",
            "userId": new_user["_id"]
        }), 201

    except Exception as e:
        print("❌ Error saving Google user:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/record-album-view", methods=["POST"])
def record_album_view():
    data = request.json
    user_id = data.get("userId")
    album_id = data.get("albumId")

    if not user_id or not album_id:
        return jsonify({"error": "Missing userId or albumId"}), 400

    # Only keep last 5 unique albums (most recent first)
    clients_collection.update_one(
        {"_id": user_id},
        {
            "$pull": {"recent_albums": album_id},  # Remove if exists
        }
    )

    clients_collection.update_one(
        {"_id": user_id},
        {
            "$push": {
                "recent_albums": {
                    "$each": [album_id],
                    "$position": 0,  # Add to beginning
                    "$slice": 5      # Keep only 5 items
                }
            }
        }
    )

    return jsonify({"message": "Album view recorded"}), 200

@app.route("/photos-from-recent-albums", methods=["POST"])
def photos_from_recent_albums():
    data = request.json
    user_id = data.get("userId")
    page = int(data.get("page", 1))
    limit = int(data.get("limit", 16))

    if not user_id:
        return jsonify({"error": "userId is required"}), 400

    user = clients_collection.find_one({"_id": user_id}, {"recent_albums": 1})
    album_ids = user.get("recent_albums", []) if user else []

    all_photos = []
    for album_id in album_ids:
        album = albums_collection.find_one({"_id": album_id})
        if album:
            all_photos.extend([
                {"photo_id": p["photo_id"], "image": p["image"]}
                for p in album.get("photos", [])
            ])

    total_photos = len(all_photos)
    skip = (page - 1) * limit
    paginated_photos = all_photos[skip:skip + limit]

    return jsonify({
        "photos": paginated_photos,
        "total": total_photos
    }), 200


@app.route("/record-download-history", methods=["POST"])
def record_download_history():
    data = request.json
    user_id = data.get("userId")
    download = data.get("download")

    if not user_id or not download:
        return jsonify({"error": "Missing userId or download object"}), 400

    download["downloadId"] = str(int(datetime.utcnow().timestamp() * 1000))
    
    # Try to get album name using the first photo URL
    image_url = download.get("photoUrls", [None])[0]
    album_name = "Downloaded Images"

    if image_url:
        album = albums_collection.find_one({"photos.image": image_url})
        if album:
            album_name = album.get("name", album_name)

    # Replace the title with the album name
    download["title"] = album_name

    # Add the new download to the beginning of the array, keep only last 10
    clients_collection.update_one(
        {"_id": user_id},
        {
            "$push": {
                "download_history": {
                    "$each": [download],
                    "$position": 0,
                    "$slice": 10
                }
            }
        }
    )

     # ✅ Increment global counters (downloads and photos)
    photo_count = download.get("photoCount", 0)
    if isinstance(photo_count, int) and photo_count > 0:
        clients_collection.update_one(
            {"_id": user_id},
            {"$inc": {
                "download_stats.downloads": 1,
                "download_stats.photos": photo_count
            }}
        )


    return jsonify({
        "message": "Download history recorded and counters updated",
        "downloadId": download["downloadId"]
    }), 200

@app.route("/get-download-history", methods=["POST"])
def get_download_history():
    data = request.json
    user_id = data.get("userId")
    
    if not user_id:
        return jsonify({"error": "Missing userId"}), 400

    user = clients_collection.find_one({"_id": user_id}, {"download_history": 1})
    history = user.get("download_history", []) if user else []

    return jsonify({"history": history}), 200


@app.route("/update-download-history", methods=["POST"])
def update_download_history():
    data = request.json
    user_id = data.get("userId")
    updated_history = data.get("updatedHistory", [])

    if not user_id:
        return jsonify({"error": "Missing userId"}), 400

    clients_collection.update_one(
        {"_id": user_id},
        {"$set": {"download_history": updated_history[:10]}}  # limit to 10
    )

    return jsonify({"message": "Download history updated"}), 200

@app.route("/update-download-date", methods=["POST"])
def update_download_date():
    data = request.json
    user_id = data.get("userId")
    download_id = data.get("downloadId")  # ✅ use ID instead of title
    new_date = data.get("date")

    if not user_id or not download_id or not new_date:
        return jsonify({"error": "Missing data"}), 400

    user = clients_collection.find_one({"_id": user_id}, {"download_history": 1})
    if not user or "download_history" not in user:
        return jsonify({"error": "User or download history not found"}), 404

    history = user["download_history"]
    updated_history = []

    for item in history:
        if str(item.get("downloadId")) == str(download_id):
            item["lastDownload"] = new_date
            updated_history.insert(0, item)  # move updated to top
        else:
            updated_history.append(item)

    # ✅ Deduplicate + limit to 10
    seen = set()
    final_history = []
    for item in updated_history:
        item_id = str(item.get("downloadId"))
        if item_id not in seen:
            final_history.append(item)
            seen.add(item_id)
        if len(final_history) == 10:
            break

    clients_collection.update_one(
        {"_id": user_id},
        {"$set": {"download_history": final_history}}
    )

    return jsonify({"message": "Download date updated and reordered"}), 200

@app.route("/get-user-download-count", methods=["POST"])
def get_user_download_count():
    data = request.json
    user_id = data.get("userId")

    if not user_id:
        return jsonify({"error": "Missing userId"}), 400

    user = clients_collection.find_one({"_id": user_id}, {"download_stats": 1})
    stats = user.get("download_stats", {}) if user else {}

    return jsonify({
        "downloads": stats.get("downloads", 0),
        "photos": stats.get("photos", 0)
    }), 200

@app.route("/total-user-downloads", methods=["GET"])
def total_user_downloads():
    try:
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total": {"$sum": "$download_stats.downloads"}
                }
            }
        ]
        result = list(clients_collection.aggregate(pipeline))
        total_downloads = result[0]["total"] if result else 0
        return jsonify({"count": total_downloads}), 200
    except Exception as e:
        print("❌ Error calculating user download sum:", e)
        return jsonify({"count": 0}), 200

@app.route("/get-user-by-email/<string:email>", methods=["GET"])
def get_user_by_email(email):
    user = clients_collection.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({
        "name": user.get("name", ""),
        "mobile": user.get("mobile", ""),
        "district": user.get("district", "")
    }), 200


@app.route("/upload-banner", methods=["POST"])
def upload_banner():
    data = request.json
    image_base64 = data.get("image")
    title = data.get("title", "Untitled Banner")
    size = data.get("size", "")

    if not image_base64 or not title:
        return jsonify({"error": "Missing title or image"}), 400

    try:
        if image_base64.startswith("data:image"):
            header, image_base64 = image_base64.split(",", 1)
            ext = header.split("/")[1].split(";")[0].lower()
            if ext not in ["png", "jpg", "jpeg"]:
                return jsonify({"error": "Unsupported image format"}), 400
        else:
            return jsonify({"error": "Invalid image data"}), 400

        filename = f"banners/{uuid.uuid4().hex}.{ext}"

        # ✅ Decode Base64 and upload to R2
        image_data = base64.b64decode(image_base64)
        s3_client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=filename,
            Body=image_data,
            ContentType=f"image/{ext}",
            ACL='public-read'
        )
        public_url = f"https://{PUBLIC_BUCKET_DOMAIN}/{filename}"

        banner_id = str(uuid.uuid4())
        banners_collection.insert_one({
            "_id": banner_id,
            "title": title,
            "image": public_url,
            "size": size,
            "date": datetime.now().strftime("%d/%m/%Y"),
        })
        return jsonify({"url": public_url, "id": banner_id}), 200

    except Exception as e:
        print("Upload error:", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/get-banners", methods=["GET"])
def get_banners():
    banners = list(dist_and_depart_db["banners"].find({}, {"_id": 1, "title": 1, "image": 1, "size": 1, "date": 1}))
    formatted = [{"id": str(b["_id"]), **b} for b in banners]
    return jsonify(formatted)


@app.route("/delete-banner/<banner_id>", methods=["DELETE"])
def delete_banner(banner_id):
    try:
        banners = dist_and_depart_db["banners"]

        # ✅ Match the string ID, not UUID object
        banner = banners.find_one({"_id": banner_id})

        if not banner:
            return jsonify({"error": "Banner not found"}), 404

        # ✅ Get R2 key from URL
        key = banner["image"].split("banners/")[1]

        # Delete from R2
        s3_client.delete_object(Bucket=R2_BUCKET_NAME, Key=f"banners/{key}")

        # Delete from DB
        banners.delete_one({"_id": banner_id})

        return jsonify({"message": "Deleted"}), 200

    except Exception as e:
        print("Error deleting banner:", e)
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))  # Get PORT from Render, default to 5000
    app.run(host="0.0.0.0", port=port, debug=False)
