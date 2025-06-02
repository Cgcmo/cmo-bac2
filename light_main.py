# main.py

# from fastapi import FastAPI, Request, UploadFile, File, Form, Depends
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import requests
from fastapi import Query
from pydantic import BaseModel
# from fastapi.responses import FileResponse
# from typing import List
# from PIL import ExifTags
# Imports same as your Flask app
import uuid
import os
# import io
# from io import BytesIO
# from PIL import Image, ImageFilter
# from rembg import new_session, remove
# from deepface import DeepFace
# import numpy as np
# from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
# from deepface.DeepFace import build_model
# from deepface.detectors import FaceDetector
import boto3
from botocore.client import Config
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi
import pymongo


# Load .env
load_dotenv()


# R2 Setup
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_REGION = os.getenv("R2_REGION")
PUBLIC_BUCKET_DOMAIN = os.getenv("PUBLIC_BUCKET_DOMAIN")
R2_ENDPOINT_URL = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

s3_client = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    config=Config(signature_version="s3v4"),
    region_name=R2_REGION,
)

client = pymongo.MongoClient(
    os.getenv("MONGO_DB_URL"),
    tls=True,
    tlsCAFile=certifi.where()
)

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

# Load Facenet model once
# print("🔧 Loading Facenet model...")
# facenet_model = build_model("Facenet")
# print("✅ Facenet model loaded.")

# print("🔧 Loading u2netp model...")
# rembg_session = new_session(model_name="u2netp")
# print("✅ model loaded.")

# print("🔧 Preloading all face embeddings grouped by photo_id...")

# photo_embeddings = {}
# photo_url_mapping = {}

# for album in albums_collection.find({}):
#     for photo in album.get("photos", []):
#         photo_id = photo.get("photo_id")
#         photo_url = photo.get("image")
#         embeddings = photo.get("face_embeddings", [])

#         if not embeddings:
#             continue
        
#         # Add photo URL mapping
#         photo_url_mapping[photo_id] = photo_url
        
#         # Add embeddings
#         face_list = []
#         for face in embeddings:
#             emb = np.array(face.get("embedding"))
#             emb_norm = np.linalg.norm(emb)  # ✅ precompute
#             face_list.append((emb, emb_norm))  # ✅ save tuple
        
#         photo_embeddings[photo_id] = face_list

# print(f"✅ Preloaded {len(photo_embeddings)} photos with embeddings into RAM")

# FastAPI app
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.options("/{full_path:path}")
async def handle_options_request(full_path: str, request: Request):
    return {"message": "CORS Preflight OK"}

# ========== Home ==========
@app.get("/")
async def home():
    return {"message": "Backend is running successfully!"}


# ========== Upload Helpers ==========

# def upload_to_r2(image_input, filename):
#     try:
#         ext = filename.split('.')[-1].lower()
#         if ext in ["jpg", "jpeg"]:
#             content_type = "image/jpeg"
#         elif ext == "png":
#             content_type = "image/png"
#         else:
#             content_type = "application/octet-stream"

#         s3_client.put_object(
#             Bucket=R2_BUCKET_NAME,
#             Key=filename,
#             Body=image_input,
#             ContentType=content_type,
#             ACL='public-read'
#         )

#         public_url = f"https://{PUBLIC_BUCKET_DOMAIN}/{filename}"
#         return public_url
#     except Exception as e:
#         print("❌ Upload to R2 failed:", str(e))
#         return None

# def delete_from_r2(file_url):
#     try:
#         if not file_url:
#             return
#         key = file_url.split(PUBLIC_BUCKET_DOMAIN + "/")[-1]
#         s3_client.delete_object(Bucket=R2_BUCKET_NAME, Key=key)
#         print(f"✅ Deleted {key} from R2")
#     except Exception as e:
#         print(f"❌ Failed to delete {file_url}: {str(e)}")


# # ========== Face Extraction Helper ==========


# def extract_faces(image_pil):
#     temp_path = None  # Initialize temp_path early for safe deletion
#     try:
#         # ✅ Step 1: Remove background
#         buffered = io.BytesIO()
#         image_pil.save(buffered, format="PNG")
#         input_bytes = buffered.getvalue()
#         output_bytes = remove(input_bytes, session=rembg_session)
#         image_no_bg = Image.open(io.BytesIO(output_bytes)).convert("RGB")

#         # ✅ Step 2: Sharpen the image
#         sharpened = image_no_bg.filter(ImageFilter.SHARPEN)

#         # ✅ Step 3: Save temporarily for DeepFace
#         temp_path = f"temp_{uuid.uuid4().hex}.jpg"
#         sharpened.save(temp_path)

#         print(f"🔍 Extracting faces from sharpened + bg-removed image: {temp_path}")

#         # ✅ Step 4: Use MTCNN for face detection
#         faces = DeepFace.represent(
#             img_path=temp_path,
#             model_name="Facenet",
#             # detector_backend="opencv",
#             enforce_detection=True
#         )

#         # ✅ Remove temp file after extraction
#         if os.path.exists(temp_path):
#             os.remove(temp_path)

#         if not faces:
#             print("❌ No faces detected.")
#             return []

        
#         # ✅ Sort faces by size and take top 4 clear faces
#         faces_sorted = sorted(
#             faces,
#             key=lambda f: f.get("facial_area", {}).get("w", 0) * f.get("facial_area", {}).get("h", 0),
#             reverse=True
#         )[:4]

#         print(f"✅ Returning top {len(faces_sorted)} faces")

#         return [
#             {
#                 "face_id": str(uuid.uuid4()),
#                 "embedding": np.array(face["embedding"]).tolist()
#             } for face in faces_sorted
#         ]

#     except Exception as e:
#         print("❌ Face extraction failed:", str(e))
#         if temp_path and os.path.exists(temp_path):
#             os.remove(temp_path)
#         return []

# ========== Create Album ==========
# @app.post("/create-album")
# async def create_album(
#     name: str = Form(...),
#     date: str = Form(...),
#     department: str = Form(""),
#     districts: str = Form(""),
#     cover: UploadFile = File(...)
# ):
#     try:
#         image = Image.open(cover.file)
#         try:
#             for orientation in ExifTags.TAGS.keys():
#                 if ExifTags.TAGS[orientation] == 'Orientation':
#                     break

#             exif = image._getexif()
#             if exif is not None:
#                 orientation_value = exif.get(orientation)
#                 if orientation_value == 3:
#                     image = image.rotate(180, expand=True)
#                 elif orientation_value == 6:
#                     image = image.rotate(270, expand=True)
#                 elif orientation_value == 8:
#                     image = image.rotate(90, expand=True)
#         except Exception as e:
#             print(f"⚠️ EXIF rotation correction failed: {e}")

#         if image.mode == "RGBA":
#             image = image.convert("RGB")

#         buffer = io.BytesIO()
#         image.save(buffer, format="JPEG", quality=50, optimize=True)
#         buffer.seek(0)
#         compressed_image = buffer.getvalue()

#         cover_filename = f"covers/{uuid.uuid4().hex}.jpg"
#         cover_url = upload_to_r2(compressed_image, cover_filename)

#         album = {
#             "_id": str(uuid.uuid4()),
#             "name": name,
#             "date": date,
#             "cover": cover_url,
#             "department": department,
#             "districts": [districts],
#             "photos": []
#         }

#         albums_collection.insert_one(album)
#         return {"message": "Album created successfully"}

#     except Exception as e:
#         print("❌ Album creation error:", str(e))
#         return JSONResponse(content={"error": "Failed to process cover image"}, status_code=500)


# # ========== Upload Photos to Gallery ==========

# @app.post("/upload-gallery/{album_id}")
# async def upload_gallery(album_id: str, photos: List[UploadFile] = File(...)):
#     try:
#         if not photos:
#             return JSONResponse(content={"error": "No files uploaded"}, status_code=400)

#         new_photos = []
#         rejected_files = []

#         for file in photos:
#             try:
#                 # ✅ Read and open the uploaded photo
#                 image_bytes = await file.read()
#                 size_kb = len(image_bytes) / 1024
#                 size_mb = size_kb / 1024
#                 print(f"📦 Received file: {file.filename} | Size: {size_kb:.2f} KB ({size_mb:.2f} MB)")

#                 image = Image.open(io.BytesIO(image_bytes))
#                 try:
#                     for orientation in ExifTags.TAGS.keys():
#                         if ExifTags.TAGS[orientation] == 'Orientation':
#                             break

#                     exif = image._getexif()
#                     if exif is not None:
#                         orientation_value = exif.get(orientation)
#                         if orientation_value == 3:
#                             image = image.rotate(180, expand=True)
#                         elif orientation_value == 6:
#                             image = image.rotate(270, expand=True)
#                         elif orientation_value == 8:
#                             image = image.rotate(90, expand=True)
#                 except Exception as e:
#                     print(f"⚠️ EXIF rotation correction failed: {e}")

                
#                 if image.mode == "RGBA":
#                     image = image.convert("RGB")

#                 # ✅ Extract face embeddings
#                 embeddings = extract_faces(image)

#                 if not embeddings:
#                     print(f"❌ No face found in: {file.filename}")
#                     rejected_files.append(file.filename)
#                     continue

#                 # ✅ Compress image
#                 buffer = io.BytesIO()
#                 image.save(buffer, format="JPEG", quality=40, optimize=True)
#                 buffer.seek(0)
#                 compressed_image = buffer.getvalue()

#                 # ✅ Upload compressed image to R2
#                 filename = f"gallery/{uuid.uuid4().hex}.jpg"
#                 image_url = upload_to_r2(compressed_image, filename)

#                 # ✅ Prepare photo record
#                 photo = {
#                     "photo_id": str(uuid.uuid4()),
#                     "image": image_url,
#                     "face_embeddings": embeddings
#                 }
#                 new_photos.append(photo)

#             except Exception as e:
#                 print(f"❌ Failed to process {file.filename}: {e}")
#                 rejected_files.append(file.filename)

#         # ✅ Update album in database
#         if new_photos:
#             albums_collection.update_one(
#                 {"_id": album_id},
#                 {"$push": {"photos": {"$each": new_photos}}}
#             )

#         return JSONResponse(content={
#             "message": "Upload complete",
#             "uploaded": len(new_photos),
#             "rejected": rejected_files
#         }, status_code=201)

#     except Exception as e:
#         print("❌ Upload failed:", str(e))
#         return JSONResponse(content={"error": "Upload failed"}, status_code=500)


# ========== Get All Albums ==========
@app.get("/albums")
async def get_albums(page: int = 1, limit: int = 16):
    try:
        skip = (page - 1) * limit

        albums = albums_collection.aggregate([
            {"$skip": skip},
            {"$limit": limit},
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

        return {
            "albums": albums,
            "total": total_count
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



class LoginRequest(BaseModel):
    identifier: str
    password: str

@app.post("/login")
async def login(data: LoginRequest):
    identifier = data.identifier
    password = data.password

    user = users_collection.find_one({"$or": [{"name": identifier}, {"mobile": identifier}]})
    
    if not user or not check_password_hash(user["password"], password):
        return JSONResponse(content={"error": "Invalid credentials"}, status_code=401)

    return {
        "message": "Login successful",
        "userId": str(user["_id"]),
        "name": user["name"],
        "mobile": user["mobile"],
        "district": user["district"]
    }


class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    district: str
    mobile: str = None

@app.post("/complete-signup")
async def complete_signup(data: SignupRequest, request: Request):
    verified_mobile = request.headers.get("X-Otpless-Mobile") or data.mobile

    if not verified_mobile:
        return JSONResponse(content={"error": "Mobile number missing or not verified"}, status_code=400)

    if clients_collection.find_one({"mobile": verified_mobile}):
        return JSONResponse(content={"error": "Mobile already registered"}, status_code=409)

    new_user = {
        "_id": str(uuid.uuid4()),
        "name": data.name,
        "email": data.email,
        "mobile": verified_mobile,
        "district": data.district,
        "role": "User",
        "status": True,
        "password": generate_password_hash(data.password)
    }

    clients_collection.insert_one(new_user)

    return {
        "message": "User registered successfully",
        "userId": new_user["_id"],
        "name": new_user["name"],
        "mobile": new_user["mobile"],
        "district": new_user["district"]
    }


class ClientLoginRequest(BaseModel):
    mobile: str
    password: str

@app.post("/client-login")
async def client_login(data: ClientLoginRequest):
    mobile = data.mobile
    password = data.password

    client = clients_collection.find_one({"mobile": mobile})

    if not client or not check_password_hash(client["password"], password):
        return JSONResponse(content={"error": "Invalid credentials"}, status_code=401)

    if not client.get("status", True):
        return JSONResponse(content={"error": "Your account is inactive. Please contact admin."}, status_code=403)

    return {
        "message": "Login successful",
        "userId": str(client["_id"]),
        "name": client["name"],
        "mobile": client["mobile"],
        "email": client.get("email", ""),
        "district": client.get("district", ""),
        "role": client.get("role", "User")
    }



# @app.post("/reload-embeddings")
# async def reload_embeddings():
#     global photo_embeddings, photo_url_mapping

#     try:
#         print("🔄 Reloading all face embeddings...")

#         photo_embeddings = {}
#         photo_url_mapping = {}

#         for album in albums_collection.find({}):
#             for photo in album.get("photos", []):
#                 photo_id = photo.get("photo_id")
#                 photo_url = photo.get("image")
#                 embeddings = photo.get("face_embeddings", [])

#                 if not embeddings:
#                     continue

#                 photo_url_mapping[photo_id] = photo_url

#                 face_list = []
#                 for face in embeddings:
#                     emb = np.array(face.get("embedding"))
#                     emb_norm = np.linalg.norm(emb)  # precompute
#                     face_list.append((emb, emb_norm))

#                 photo_embeddings[photo_id] = face_list

#         print(f"✅ Reloaded {len(photo_embeddings)} photos into RAM")

#         return {"message": "Embeddings reloaded successfully", "count": len(photo_embeddings)}

#     except Exception as e:
#         print("❌ Error reloading embeddings:", str(e))
#         return JSONResponse(content={"error": str(e)}, status_code=500)



# @app.post("/search-by-upload")
# async def search_by_upload(image: UploadFile = File(...)):
#     if not image:
#         return JSONResponse(content={"error": "No image file provided"}, status_code=400)

#     if image.filename == "":
#         return JSONResponse(content={"error": "No file selected"}, status_code=400)

#     try:
#         # ✅ Convert uploaded file to PIL Image
#         img_bytes = await image.read()
#         image_obj = Image.open(BytesIO(img_bytes))
#         try:
#             for orientation in ExifTags.TAGS.keys():
#                 if ExifTags.TAGS[orientation] == 'Orientation':
#                     break

#             exif = image_obj._getexif()
#             if exif is not None:
#                 orientation_value = exif.get(orientation)
#                 if orientation_value == 3:
#                     image_obj = image_obj.rotate(180, expand=True)
#                 elif orientation_value == 6:
#                     image_obj = image_obj.rotate(270, expand=True)
#                 elif orientation_value == 8:
#                     image_obj = image_obj.rotate(90, expand=True)

#         except Exception as e:
#             print(f"⚠️ EXIF rotation correction failed: {e}")

#         image_obj.load()  # force decoding
#         file_size_kb = len(img_bytes) / 1024  # in KB
#         file_size_kb = round(file_size_kb, 2)  # rounded to 2 decimals

#         print(f"✅ Uploaded image file size: {file_size_kb} KB")
#         print("✅ Uploaded image format:", image_obj.format, "| size:", image_obj.size, "| mode:", image_obj.mode)

#         if image_obj.mode == "RGBA":
#             image_obj = image_obj.convert("RGB")

#         # ✅ Extract faces from uploaded image
#         query_embeddings = extract_faces(image_obj)

#         if not query_embeddings:
#             return JSONResponse(content={"error": "No face found in uploaded photo"}, status_code=404)

#         matched_photo_ids = set()

#         # ✅ Step 3: Compare with preloaded photo_embeddings instead of MongoDB
#         for query_face in query_embeddings:
#             query_emb = np.array(query_face["embedding"])
            
#             query_norm = np.linalg.norm(query_emb)  # Only once per query face

#             for query_face in query_embeddings:
#                 query_emb = np.array(query_face["embedding"])
#                 query_norm = np.linalg.norm(query_emb)  # ✅ once

#                 for photo_id, faces in photo_embeddings.items():
#                     for emb, emb_norm in faces:  # ✅ get both
#                         cosine_sim = np.dot(query_emb, emb) / (query_norm * emb_norm)
#                         if cosine_sim > 0.75:
#                             matched_photo_ids.add(photo_id)
#                             break

#         if not matched_photo_ids:
#             return JSONResponse(
#                 content={"error": "No matching faces found in database, either there is no photo of uploaded face or face is not clear"},
#                 status_code=404
#             )

#         # ✅ Step 4: Prepare the matched photos to return
#         matched_photos = [{"photo_id": pid, "image": photo_url_mapping[pid]} for pid in matched_photo_ids]

#         return {"photos": matched_photos}

#     except Exception as e:
#         print("❌ Error in /search-by-upload:", str(e))
#         return JSONResponse(content={"error": str(e)}, status_code=500)



@app.post("/record-visit")
async def record_visit():
    try:
        visitor_collection.insert_one({
            "timestamp": datetime.utcnow()
        })
        return {"message": "Visit recorded"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/visitor-stats")
async def visitor_stats():
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

        count_map = {
            f"{d['_id']['year']}-{d['_id']['month']:02d}-{d['_id']['day']:02d}": d["count"]
            for d in raw_data
        }

        results = []
        for day in last_7_days:
            key = day.strftime("%Y-%m-%d")
            results.append({
                "date": key,
                "count": count_map.get(key, 0)
            })

        return results
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



# @app.post("/upload-banner")
# async def upload_banner(
#     title: str = Form(...),
#     size: str = Form(""),
#     image: UploadFile = File(...)
# ):
#     if not image or not title:
#         return JSONResponse(content={"error": "Missing title or image"}, status_code=400)

#     try:
#         content_type = image.content_type
#         ext = content_type.split("/")[-1].lower()

#         if ext not in ["png", "jpg", "jpeg"]:
#             return JSONResponse(content={"error": "Unsupported image format"}, status_code=400)

#         filename = f"banners/{uuid.uuid4().hex}.{ext}"
#         image_bytes = await image.read()

#         public_url = upload_to_r2(image_bytes, filename)

#         banner_id = str(uuid.uuid4())
#         banners_collection.insert_one({
#             "_id": banner_id,
#             "title": title,
#             "image": public_url,
#             "size": size,
#             "date": datetime.now().strftime("%d/%m/%Y"),
#         })

#         return {"url": public_url, "id": banner_id}

#     except Exception as e:
#         print("Upload error:", e)
#         return JSONResponse(content={"error": "Internal server error"}, status_code=500)


@app.get("/get-banners")
async def get_banners():
    banners = list(dist_and_depart_db["banners"].find({}, {"_id": 1, "title": 1, "image": 1, "size": 1, "date": 1}))
    formatted = [{"id": str(b["_id"]), **b} for b in banners]
    return formatted



# @app.delete("/delete-banner/{banner_id}")
# async def delete_banner(banner_id: str):
#     try:
#         banners = dist_and_depart_db["banners"]
#         banner = banners.find_one({"_id": banner_id})

#         if not banner:
#             return JSONResponse(content={"error": "Banner not found"}, status_code=404)

#         key = banner["image"].split("banners/")[1]

#         s3_client.delete_object(Bucket=R2_BUCKET_NAME, Key=f"banners/{key}")

#         banners.delete_one({"_id": banner_id})

#         return {"message": "Deleted"}

#     except Exception as e:
#         print("Error deleting banner:", e)
#         return JSONResponse(content={"error": "Internal server error"}, status_code=500)


@app.get("/albums-by-district")
async def get_albums_by_district(name: str, page: int = 1, limit: int = 16):
    if not name:
        return JSONResponse(content={"error": "District name is required"}, status_code=400)

    try:
        skip = (page - 1) * limit
        query = {"districts": {"$in": [name]}}
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
                "photo_count": {"$size": {"$ifNull": ["$photos", []]}}
            }}
        ]))

        return {
            "albums": albums,
            "total": total
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


class AlbumDeleteRequest(BaseModel):
    albumIds: list[str]

@app.delete("/delete-albums")
async def delete_multiple_albums(data: AlbumDeleteRequest):
    album_ids = data.albumIds

    if not album_ids:
        return JSONResponse(content={"error": "No album IDs provided"}, status_code=400)

    try:
        for album_id in album_ids:
            album = albums_collection.find_one({"_id": album_id})
            if album:
                delete_from_r2(album.get("cover"))
                for photo in album.get("photos", []):
                    delete_from_r2(photo.get("image"))

        result = albums_collection.delete_many({"_id": {"$in": album_ids}})
        return {"message": f"Deleted {result.deleted_count} albums successfully"}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



# @app.delete("/photo/{album_id}/{photo_id}")
# async def delete_photo(album_id: str, photo_id: str):
#     album = albums_collection.find_one({"_id": album_id}, {"photos": 1})
#     if not album:
#         return JSONResponse(content={"error": "Album not found"}, status_code=404)

#     photo_to_delete = next((photo for photo in album.get("photos", []) if photo.get("photo_id") == photo_id), None)

#     if not photo_to_delete:
#         return JSONResponse(content={"error": "Photo not found"}, status_code=404)

#     delete_from_r2(photo_to_delete.get("image"))

#     albums_collection.update_one({"_id": album_id}, {"$pull": {"photos": {"photo_id": photo_id}}})

#     return {"message": "Photo deleted successfully"}


@app.get("/get-events")
async def get_events():
    events = albums_collection.find({}, {"name": 1, "_id": 0})
    event_names = [event["name"] for event in events]
    return {"events": event_names}


class EventRequest(BaseModel):
    eventName: str

@app.post("/fetch-album-photos")
async def fetch_album_photos(data: EventRequest):
    event_name = data.eventName

    if not event_name:
        return JSONResponse(content={"error": "Event name is required"}, status_code=400)

    album = albums_collection.find_one({"name": event_name})

    if not album:
        return JSONResponse(content={"error": "No album found with this name"}, status_code=404)

    photos = album.get("photos", [])

    return {
        "photos": [{"photo_id": photo["photo_id"], "image": photo["image"]} for photo in photos]
    }


class DateRequest(BaseModel):
    date: str

@app.post("/fetch-photos-by-date")
async def fetch_photos_by_date(data: DateRequest):
    selected_date = data.date

    if not selected_date:
        return JSONResponse(content={"error": "Date is required"}, status_code=400)

    albums = albums_collection.find({"date": selected_date})
    all_photos = []

    for album in albums:
        all_photos.extend([
            {"photo_id": photo["photo_id"], "image": photo["image"]}
            for photo in album.get("photos", [])
        ])

    if not all_photos:
        return JSONResponse(content={"error": "No photos found for this date"}, status_code=404)

    return {"photos": all_photos}

class MasterSearchRequest(BaseModel):
    query: str

@app.post("/master-search")
async def master_search(data: MasterSearchRequest):
    query = data.query.strip().lower()

    if not query:
        return JSONResponse(content={"error": "Empty search"}, status_code=400)

    matching_photos = []

    albums = albums_collection.find()

    for album in albums:
        # Simple case-insensitive matching
        album_name = album.get("name", "").lower()
        department = album.get("department", "").lower()
        districts = [d.lower() for d in album.get("districts", [])]

        if (
            query in album_name or
            query in department or
            any(query in d for d in districts)
        ):
            matched_by = []
            if query in album_name:
                matched_by.append("Event")
            if query in department:
                matched_by.append("Department")
            if any(query in d for d in districts):
                matched_by.append("District")

            for photo in album.get("photos", []):
                matching_photos.append({
                    "photo_id": photo["photo_id"],
                    "image": photo["image"],
                    "matched_by": matched_by,
                    "album_name": album.get("name", ""),
                    "department": album.get("department", ""),
                    "districts": album.get("districts", [])
                })

    return {"photos": matching_photos}


# class SuggestionRequest(BaseModel):
#     partialQuery: str
@app.get("/search-suggestions")
async def search_suggestions():
    events = [e["name"] for e in albums_collection.find({}, {"name": 1})]
    departments = [d["name"] for d in departments_collection.find({}, {"name": 1})]
    districts = [d["name"] for d in districts_collection.find({}, {"name": 1})]

    return {
        "events": list(set(events)),
        "departments": list(set(departments)),
        "districts": list(set(districts))
    }




@app.get("/fetch-all-photos")
async def fetch_all_photos(page: int = 1, limit: int = 16):
    try:
        skip = (page - 1) * limit

        albums = albums_collection.find({}, {"photos": 1})
        all_photos = []

        for album in albums:
            for photo in album.get("photos", []):
                all_photos.append({
                    "photo_id": photo["photo_id"],
                    "image": photo["image"]
                })

        paginated_photos = all_photos[skip:skip+limit]

        return {
            "photos": paginated_photos,
            "total": len(all_photos)
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


class GoogleLoginRequest(BaseModel):
    email: str
    name: str = None
    photo: str = None

@app.post("/google-login")
async def google_login(data: GoogleLoginRequest):
    email = data.email

    if not email:
        return JSONResponse(content={"error": "Email is required"}, status_code=400)

    existing_user = clients_collection.find_one({"email": email})
    
    if existing_user:
        if not existing_user.get("status", True):
            return JSONResponse(content={"error": "Your account is inactive. Please contact admin."}, status_code=403)
        return {
            "message": "User already exists",
            "userId": existing_user["_id"]
        }

    try:
        photo = data.photo  # ✅ Can be None if not provided

        new_user = {
            "_id": str(uuid.uuid4()),
            "name": data.name,
            "email": email,
            "mobile": "",
            "district": "",
            "role": "User",
            "status": True,
            "password": "",
        }

        if photo:  # ✅ Only add if photo provided
            new_user["photo"] = photo

        clients_collection.insert_one(new_user)

        return JSONResponse(content={
            "message": "Google user registered",
            "userId": new_user["_id"]
        }, status_code=201)

    except Exception as e:
        print("❌ Error saving Google user:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)


class CheckUserExistsRequest(BaseModel):
    mobile: str
    email: str

@app.post("/check-user-exists")
async def check_user_exists(data: CheckUserExistsRequest):
    mobile = data.mobile
    email = data.email

    if not mobile or not email:
        return JSONResponse(content={"error": "Mobile and Email required"}, status_code=400)

    # Check for mobile
    if clients_collection.find_one({"mobile": mobile}):
        return JSONResponse(content={"error": "Mobile number already registered"}, status_code=409)

    # Check for email
    if clients_collection.find_one({"email": email}):
        return JSONResponse(content={"error": "Email already registered"}, status_code=409)

    return {"message": "Mobile and Email are available"}



class ResetPasswordRequest(BaseModel):
    mobile: str
    newPassword: str

@app.post("/reset-password")
async def reset_password(data: ResetPasswordRequest):
    mobile = data.mobile
    new_password = data.newPassword

    result = clients_collection.update_one(
        {"mobile": mobile},
        {"$set": {"password": generate_password_hash(new_password)}}
    )

    if result.modified_count == 0:
        return JSONResponse(content={"error": "Failed to reset password"}, status_code=400)

    return {"message": "Password reset successful"}


# class AlbumViewRequest(BaseModel):
#     album_id: str

class RecordAlbumViewRequest(BaseModel):
    userId: str
    albumId: str

@app.post("/record-album-view")
async def record_album_view(data: RecordAlbumViewRequest):
    user_id = data.userId
    album_id = data.albumId

    if not user_id or not album_id:
        return JSONResponse(content={"error": "Missing userId or albumId"}, status_code=400)

    # ✅ Remove existing albumId if exists
    clients_collection.update_one(
        {"_id": user_id},
        {
            "$pull": {"recent_albums": album_id}
        }
    )

    # ✅ Add albumId to beginning, limit to 5 items
    clients_collection.update_one(
        {"_id": user_id},
        {
            "$push": {
                "recent_albums": {
                    "$each": [album_id],
                    "$position": 0,
                    "$slice": 5
                }
            }
        }
    )

    return {"message": "Album view recorded"}


class RecentAlbumsRequest(BaseModel):
    userId: str
    page: int = 1
    limit: int = 16

@app.post("/photos-from-recent-albums")
async def photos_from_recent_albums(data: RecentAlbumsRequest):
    user_id = data.userId
    page = data.page
    limit = data.limit

    if not user_id:
        return JSONResponse(content={"error": "userId is required"}, status_code=400)

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

    return {
        "photos": paginated_photos,
        "total": total_photos
    }

# class DownloadRecordRequest(BaseModel):
#     userId: str
#     title: str
#     date: str
#     image: str
#     photoCount: int

class RecordDownloadHistoryRequest(BaseModel):
    userId: str
    download: dict

@app.post("/record-download-history")
async def record_download_history(data: RecordDownloadHistoryRequest):
    user_id = data.userId
    download = data.download

    if not user_id or not download:
        return JSONResponse(content={"error": "Missing userId or download object"}, status_code=400)

    # ✅ Generate downloadId based on timestamp
    download["downloadId"] = str(int(datetime.utcnow().timestamp() * 1000))

    # ✅ Try to find album name using first photo URL
    image_url = download.get("photoUrls", [None])[0]
    album_name = "Downloaded Images"

    if image_url:
        album = albums_collection.find_one({"photos.image": image_url})
        if album:
            album_name = album.get("name", album_name)

    # ✅ Update download title with album name
    download["title"] = album_name

    # ✅ Add new download to user's history, keeping only last 10
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

    # ✅ Increment user's global download and photo counters
    photo_count = download.get("photoCount", 0)
    if isinstance(photo_count, int) and photo_count > 0:
        clients_collection.update_one(
            {"_id": user_id},
            {"$inc": {
                "download_stats.downloads": 1,
                "download_stats.photos": photo_count
            }}
        )

    return {
        "message": "Download history recorded and counters updated",
        "downloadId": download["downloadId"]
    }


class GetDownloadHistoryRequest(BaseModel):
    userId: str

@app.post("/get-download-history")
async def get_download_history(data: GetDownloadHistoryRequest):
    user_id = data.userId

    if not user_id:
        return JSONResponse(content={"error": "Missing userId"}, status_code=400)

    user = clients_collection.find_one({"_id": user_id}, {"download_history": 1})
    history = user.get("download_history", []) if user else []

    return {
        "history": history
    }


class UpdateDownloadDateRequest(BaseModel):
    userId: str
    downloadId: str
    date: str

@app.post("/update-download-date")
async def update_download_date(data: UpdateDownloadDateRequest):
    user_id = data.userId
    download_id = data.downloadId
    new_date = data.date

    if not user_id or not download_id or not new_date:
        return JSONResponse(content={"error": "Missing data"}, status_code=400)

    user = clients_collection.find_one({"_id": user_id}, {"download_history": 1})
    if not user or "download_history" not in user:
        return JSONResponse(content={"error": "User or download history not found"}, status_code=404)

    history = user["download_history"]
    updated_history = []

    for item in history:
        if str(item.get("downloadId")) == str(download_id):
            item["lastDownload"] = new_date
            updated_history.insert(0, item)  # ✅ Move updated one to top
        else:
            updated_history.append(item)

    # ✅ Deduplicate + Limit to 10
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

    return {"message": "Download date updated and reordered"}


@app.get("/total-user-downloads")
async def total_user_downloads():
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
        return {"count": total_downloads}
    except Exception as e:
        print("❌ Error calculating user download sum:", e)
        return {"count": 0}

@app.get("/get-user-by-email/{email}")
async def get_user_by_email(email: str):
    user = clients_collection.find_one({"email": email})

    if not user:
        return JSONResponse(content={"error": "User not found"}, status_code=404)

    return {
        "name": user.get("name", ""),
        "mobile": user.get("mobile", ""),
        "district": user.get("district", ""),
        "photo": user.get("photo", "")  # ✅ Added to send profile picture URL if available
    }


@app.get("/proxy-image")
async def proxy_image(url: str):
    try:
        response = requests.get(url, stream=True, timeout=10)

        if response.status_code != 200:
            return JSONResponse(content={"error": "Failed to fetch image"}, status_code=400)

        return StreamingResponse(response.raw, media_type="image/jpeg")

    except Exception as e:
        print("❌ Proxy image error:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

@app.get("/photos/{album_id}")
async def get_album_photos(album_id: str, page: int = Query(1), limit: int = Query(16)):
    try:
        album = albums_collection.find_one({"_id": album_id})  # ✅ First fetch album
        if not album:
            return JSONResponse(content={"error": "Album not found"}, status_code=404)

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

        return {
            "photos": formatted_photos,
            "total": total_photos
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.delete("/album/{album_id}")
async def delete_album(album_id: str):
    album = albums_collection.find_one({"_id": album_id})
    if not album:
        return JSONResponse(content={"error": "Album not found"}, status_code=404)

    # ✅ Delete cover image from R2
    delete_from_r2(album.get("cover"))

    # ✅ Delete all photos from R2
    for photo in album.get("photos", []):
        delete_from_r2(photo.get("image"))

    # ✅ Now delete from MongoDB
    albums_collection.delete_one({"_id": album_id})

    return {"message": "Album and its photos deleted successfully"}


@app.get("/districts")
async def get_districts():
    districts = list(districts_collection.find({}, {"_id": 0}))
    return districts


class DistrictRequest(BaseModel):
    name: str

@app.post("/districts")
async def add_district(data: DistrictRequest):
    if not data.name:
        return JSONResponse(content={"error": "District name is required"}, status_code=400)
    districts_collection.insert_one({"name": data.name})
    return {"message": "District added successfully"}

class UpdateDistrictRequest(BaseModel):
    name: str

@app.put("/districts/{old_name}")
async def edit_district(old_name: str, data: UpdateDistrictRequest):
    if not data.name:
        return JSONResponse(content={"error": "New district name is required"}, status_code=400)
    districts_collection.update_one({"name": old_name}, {"$set": {"name": data.name}})
    return {"message": "District updated successfully"}


@app.delete("/districts/{name}")
async def delete_district(name: str):
    districts_collection.delete_one({"name": name})
    return {"message": "District deleted successfully"}


@app.get("/departments")
async def get_departments():
    departments = list(departments_collection.find({}, {"_id": 0}))
    return departments


class DepartmentRequest(BaseModel):
    name: str

@app.post("/departments")
async def add_department(data: DepartmentRequest):
    if not data.name:
        return JSONResponse(content={"error": "Department name is required"}, status_code=400)
    departments_collection.insert_one({"name": data.name})
    return {"message": "Department added successfully"}


class UpdateDepartmentRequest(BaseModel):
    name: str

@app.put("/departments/{old_name}")
async def edit_department(old_name: str, data: UpdateDepartmentRequest):
    if not data.name:
        return JSONResponse(content={"error": "New department name is required"}, status_code=400)
    departments_collection.update_one({"name": old_name}, {"$set": {"name": data.name}})
    return {"message": "Department updated successfully"}


@app.delete("/departments/{name}")
async def delete_department(name: str):
    departments_collection.delete_one({"name": name})
    return {"message": "Department deleted successfully"}


class StaffRequest(BaseModel):
    name: str
    email: str
    mobile: str
    password: str
    district: str

@app.post("/add-staff")
async def add_staff(data: StaffRequest):
    # Hash the password
    hashed_password = generate_password_hash(data.password)

    new_user = {
        "_id": str(uuid.uuid4()),
        "name": data.name,
        "email": data.email,
        "mobile": data.mobile,
        "district": data.district,
        "role": "Admin",  # Default role
        "password": hashed_password,
        "status": True
    }

    users_collection.insert_one(new_user)

    return {"message": "Staff added successfully"}


class UpdateUserRequest(BaseModel):
    name: str = None
    email: str = None
    mobile: str = None
    district: str = None
    status: bool = None

@app.put("/update-user/{user_id}")
async def update_user(user_id: str, data: UpdateUserRequest):
    update_fields = {}

    for field in ["name", "email", "mobile", "district", "status"]:
        value = getattr(data, field)
        if value is not None:
            update_fields[field] = value

    if not update_fields:
        return JSONResponse(content={"error": "No fields to update"}, status_code=400)

    result = users_collection.update_one({"_id": user_id}, {"$set": update_fields})

    if result.modified_count == 0:
        result = clients_collection.update_one({"_id": user_id}, {"$set": update_fields})
        if result.modified_count == 0:
            return JSONResponse(content={"error": "User not found or no changes made"}, status_code=404)

    return {"message": "User updated successfully"}


@app.get("/users")
async def get_users():
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

    return combined_users



# @app.get("/uploads/{filename}")
# async def serve_photo(filename: str):
#     file_path = f"uploads/{filename}"
#     return FileResponse(file_path, media_type="image/jpeg")


@app.get("/count-albums")
async def count_albums():
    count = albums_collection.count_documents({})
    return {"total_albums": count}

@app.get("/count-photos")
async def count_photos():
    total_photos = 0
    for album in albums_collection.find({}, {"photos": 1}):
        total_photos += len(album.get("photos", []))
    return {"total_photos": total_photos}

@app.get("/count-users")
async def count_users():
    try:
        user_count = users_collection.count_documents({})
        client_count = clients_collection.count_documents({})
        total = user_count + client_count
        return {"total_users": total}
    except Exception as e:
        print("❌ Error in /count-users:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/increment-download-count")
async def increment_download_count():
    try:
        result = download_count_collection.find_one({"_id": "total"})
        if result:
            download_count_collection.update_one({"_id": "total"}, {"$inc": {"count": 1}})
        else:
            download_count_collection.insert_one({"_id": "total", "count": 1})
        return {"message": "Download count updated"}
    except Exception as e:
        print("❌ Error incrementing download count:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


class UpdateClientRequest(BaseModel):
    name: str = None
    mobile: str = None
    district: str = None

@app.put("/update-client/{user_id}")
async def update_client(user_id: str, data: UpdateClientRequest):
    allowed_fields = ["name", "mobile", "district"]

    update_fields = {}

    for field in allowed_fields:
        value = getattr(data, field)
        if value is not None:
            update_fields[field] = value

    if not update_fields:
        return JSONResponse(content={"error": "No valid fields to update"}, status_code=400)

    result = clients_collection.update_one({"_id": user_id}, {"$set": update_fields})

    if result.modified_count == 0:
        return JSONResponse(content={"error": "Client not found or no changes made"}, status_code=404)

    return {"message": "Client updated successfully"}


class UpdateDownloadHistoryRequest(BaseModel):
    userId: str
    updatedHistory: list = []

@app.post("/update-download-history")
async def update_download_history(data: UpdateDownloadHistoryRequest):
    user_id = data.userId
    updated_history = data.updatedHistory

    if not user_id:
        return JSONResponse(content={"error": "Missing userId"}, status_code=400)

    clients_collection.update_one(
        {"_id": user_id},
        {"$set": {"download_history": updated_history[:10]}}  # ✅ Limit to 10 items
    )

    return {"message": "Download history updated"}



class GetUserDownloadCountRequest(BaseModel):
    userId: str

@app.post("/get-user-download-count")
async def get_user_download_count(data: GetUserDownloadCountRequest):
    user_id = data.userId

    if not user_id:
        return JSONResponse(content={"error": "Missing userId"}, status_code=400)

    user = clients_collection.find_one({"_id": user_id}, {"download_stats": 1})
    stats = user.get("download_stats", {}) if user else {}

    return {
        "downloads": stats.get("downloads", 0),
        "photos": stats.get("photos", 0)
    }



