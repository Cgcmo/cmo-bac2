
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
    

def extract_faces(image_data):
    image_path = f"temp_{uuid.uuid4().hex}.jpg"
    with open(image_path, "wb") as f:
        f.write(base64.b64decode(image_data))
    try:
        print(f"🔍 Extracting faces from: {image_path}")

        faces = DeepFace.represent(
            img_path=image_path,
            model_name="SFace",  # ✅ Lightweight model to avoid memory issues
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


@app.route("/get-events", methods=["GET"])
def get_events():
    events = albums_collection.find({}, {"name": 1, "_id": 0})  # Use the correct collection & field name
    event_names = [event["name"] for event in events]  # Extract event names
    return jsonify(event_names)

@app.route("/uploads/<filename>")
def serve_photo(filename):
    return send_file(f"uploads/{filename}", mimetype="image/jpeg")


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



if __name__ == "__main__":
    app.run(debug=True, port=5001)
