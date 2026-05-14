# from fastapi import FastAPI, Request, UploadFile, File, Form, Depends
# from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse, StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import uvicorn
# import requests
# from fastapi import Query
# from pydantic import BaseModel
# from fastapi.responses import FileResponse
# from typing import List
# from PIL import ExifTags
# import uuid
# import os
# import io
# from io import BytesIO
# from PIL import Image, ImageFilter
# from rembg import new_session, remove
# from deepface import DeepFace
# import numpy as np
# from bson.objectid import ObjectId
# from werkzeug.security import generate_password_hash, check_password_hash
# from datetime import datetime, timedelta
# from deepface.DeepFace import build_model
# # from deepface.detectors import FaceDetector
# import boto3
# from botocore.client import Config
# import requests
# from dotenv import load_dotenv
# from pymongo import MongoClient
# import certifi
# import pymongo
# from fastapi import Form
# from fastapi import UploadFile, File
# import asyncio
# from fastapi import HTTPException
# from concurrent.futures import ThreadPoolExecutor


# # Load .env
# load_dotenv()


# # R2 Setup
# R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
# R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
# R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
# R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
# R2_REGION = os.getenv("R2_REGION")
# PUBLIC_BUCKET_DOMAIN = os.getenv("PUBLIC_BUCKET_DOMAIN")
# R2_ENDPOINT_URL = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# s3_client = boto3.client(
#     's3',
#     endpoint_url=R2_ENDPOINT_URL,
#     aws_access_key_id=R2_ACCESS_KEY_ID,
#     aws_secret_access_key=R2_SECRET_ACCESS_KEY,
#     config=Config(signature_version="s3v4"),
#     region_name=R2_REGION,
# )

# client = pymongo.MongoClient(
#     os.getenv("MONGO_DB_URL"),
#     tls=True,
#     tlsCAFile=certifi.where()
# )

# photo_gallery_db = client["photo_gallery"]
# albums_collection = photo_gallery_db["albums"]

# dist_and_depart_db = client["dist_and_depart"]
# districts_collection = dist_and_depart_db["districts"]
# departments_collection = dist_and_depart_db["departments"]
# banners_collection = dist_and_depart_db["banners"]

# auth_db = client["auth_db"]
# users_collection = auth_db["users"]
# clients_collection = auth_db["clients"]
# download_count_collection = auth_db["download-count"]
# visitor_collection = auth_db["visitor_logs"]

# # Load Facenet model once
# print("🔧 Loading Facenet model...")
# facenet_model = build_model("Facenet")
# print("✅ Facenet model loaded.")

# print("🔧 Loading u2netp model...")
# rembg_session = new_session(model_name="u2netp")
# print("✅ model loaded.")

# print("🔧 Preloading all face embeddings grouped by photo_id (max 100k embeddings)...")

# photo_embeddings = {}
# photo_url_mapping = {}
# embedding_counter = 0
# MAX_EMBEDDINGS = 100_000

# # Sort albums by last_updated descending (newest first)
# for album in albums_collection.find({}).sort("last_updated", -1):
#     for photo in album.get("photos", []):
#         photo_id = photo.get("photo_id")
#         photo_url = photo.get("image")
#         embeddings = photo.get("face_embeddings", [])

#         if not embeddings:
#             continue

#         # Check if adding these embeddings would exceed the limit
#         if embedding_counter + len(embeddings) > MAX_EMBEDDINGS:
#             print(f"⚠️ Stopping preload after {embedding_counter} embeddings loaded (limit reached).")
#             break

#         # Add photo URL mapping
#         photo_url_mapping[photo_id] = photo_url

#         # Add embeddings
#         face_list = []
#         for face in embeddings:
#             emb = np.array(face.get("embedding"))
#             emb_norm = np.linalg.norm(emb)  # ✅ precompute
#             face_list.append((emb, emb_norm))  # ✅ save tuple
#             embedding_counter += 1

#         photo_embeddings[photo_id] = face_list

#     # Double-break if limit is hit inside inner loop
#     if embedding_counter >= MAX_EMBEDDINGS:
#         break

# print(f"✅ Preloaded {len(photo_embeddings)} photos and {embedding_counter} embeddings into RAM")

# # FastAPI app
# app = FastAPI()
# MAX_UPLOAD_QUEUE = 22
# upload_gallery_semaphore = asyncio.BoundedSemaphore(MAX_UPLOAD_QUEUE)
# gallery_executor = ThreadPoolExecutor(max_workers=1) 
# face_extract_executor = ThreadPoolExecutor(max_workers=2)
# request_semaphore = asyncio.BoundedSemaphore(666)

# # CORS Middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# @app.options("/{full_path:path}")
# async def handle_options_request(full_path: str, request: Request):
#     return {"message": "CORS Preflight OK"}

# # ========== Home ==========
# @app.get("/")
# async def home():
#     return {"message": "Backend is running successfully!!!!!"}


# # ========== Upload Helpers ==========

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


# # # ========== Face Extraction Helper ==========

# def extract_faces(image_pil):
#     temp_path = None  # Initialize temp_path early for safe deletion
#     try:
#         # Step 1: Save image temporarily
#         temp_path = f"temp_{uuid.uuid4().hex}.jpg"
#         image_pil.save(temp_path)

#         print(f"🔍 Extracting faces using MTCNN from image: {temp_path}")

#         # Step 2: Detect faces + Extract embeddings
#         faces = DeepFace.represent(
#             img_path=temp_path,
#             model_name="Facenet",
#             detector_backend="retinaface",  # 🔥 MTCNN Detector
#             enforce_detection=True
#         )

#         if not faces:
#             print("❌ No faces detected.")
#             return [],  "NoFaceDetected"

#         # Step 3: Filter faces with resolution >= 300px
#         filtered_faces = []
#         min_size = 180
#         tolerance = 50

#         for face in faces:
#             area = face.get("facial_area", {})
#             w, h = area.get("w", 0), area.get("h", 0)

#             if w >= min_size and h >= min_size:
#                 # ✅ Both sides >= 200px
#                 filtered_faces.append(face)
#             elif w < min_size and h < min_size:
#                 # ❌ Both sides < 200px
#                 print(f"⚠️ Skipping face {w}x{h} — both sides < {min_size}px")
#                 continue
#             else:
#                 # One side >= 200px, check difference
#                 diff = abs(w - h)
#                 if diff <= tolerance:
#                     filtered_faces.append(face)  # ✅ Accept
#                 else:
#                     print(f"⚠️ Skipping face {w}x{h} — size difference {diff}px > {tolerance}px")

#         if not filtered_faces:
#             print("❌ No faces with required resolution.")
#             return [], "LowResolution"

#         # Step 4: Sort filtered faces by size (area w*h) and pick top 4
#         faces_sorted = sorted(
#             filtered_faces,
#             key=lambda f: f.get("facial_area", {}).get("w", 0) * f.get("facial_area", {}).get("h", 0),
#             reverse=True
#         )[:4]

#         output_faces = []

#         # Step 5: (Optional) Crop faces for preview/logging
#         with Image.open(temp_path) as original:
#             for face in faces_sorted:
#                 area = face.get("facial_area", {})
#                 x, y, w, h = area.get("x", 0), area.get("y", 0), area.get("w", 0), area.get("h", 0)

#                 cropped = original.crop((x, y, x + w, y + h))

#                 buffer = io.BytesIO()
#                 cropped.save(buffer, format="JPEG", quality=90)
#                 size_kb = len(buffer.getvalue()) / 1024

#                 print(f"🖼️ Face: {w}x{h} pixels | {round(size_kb, 2)} KB")

#                 output_faces.append({
#                     "face_id": str(uuid.uuid4()),
#                     "embedding": np.array(face["embedding"]).tolist(),  # ✅ Embedding already extracted
#                     # (Optional: Could also save cropped image if needed)
#                 })

#         print(f"✅ Returning {len(output_faces)} faces with size/resolution.")

#         return output_faces, None

#     except Exception as e:
#         print("❌ Face extraction failed:", str(e))
#         return [], "ExtractionError"

#     finally:
#         # Clean up temp file
#         if temp_path and os.path.exists(temp_path):
#             os.remove(temp_path)


# # ========== Create Album ==========
# @app.post("/create-album")
# async def create_album(
#     name: str = Form(...),
#     date: str = Form(...),
#     department: str = Form(""),
#     districts: str = Form(""),
#     with_cm: str = Form("without"),
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
#         now = datetime.utcnow()
#         album = {
#             "_id": str(uuid.uuid4()),
#             "name": name,
#             "date": date,
#             "cover": cover_url,
#             "department": department,
#             "districts": [districts],
#              "with_cm": with_cm,
#             "photos": [],
#             "last_updated": now 
#         }

#         albums_collection.insert_one(album)
#         return {"message": "Album created successfully"}

#     except Exception as e:
#         print("❌ Album creation error:", str(e))
#         return JSONResponse(content={"error": "Failed to process cover image"}, status_code=500)


# # # ========== Upload Photos to Gallery ==========

# @app.post("/upload-gallery/{album_id}")
# async def upload_gallery(album_id: str, photos: List[UploadFile] = File(...)):
#     try:
#         await upload_gallery_semaphore.acquire()
#     except ValueError:
#         raise HTTPException(status_code=503, detail="Server busy. Try again later.")

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
#                 loop = asyncio.get_running_loop()
#                 embeddings, extraction_error = await loop.run_in_executor(gallery_executor, extract_faces, image)

#                 if extraction_error == "LowResolution" or extraction_error == "NoFaceDetected":
#                     print(f"❌ Face extraction failed for {file.filename} due to {extraction_error}")
                    # rejected_files.append(file.filename)
#                     continue

#                 if not embeddings:
#                     print(f"❌ No embeddings generated for {file.filename}")
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
#             now = datetime.utcnow()
#             albums_collection.update_one(
#                 {"_id": album_id},
#                 {"$push": {"photos": {"$each": new_photos}},
#                  "$set": {"last_updated": now} 
#                 }
#             )

#         return JSONResponse(content={
#             "message": "Upload complete",
#             "uploaded": len(new_photos),
#             "rejected": rejected_files
#         }, status_code=201)

#     except Exception as e:
#         print("❌ Upload failed:", str(e))
#         return JSONResponse(content={"error": "Upload failed"}, status_code=500)
    
#     finally:
#         upload_gallery_semaphore.release()

# @app.post("/reload-embeddings")
# async def reload_embeddings():
#     global photo_embeddings, photo_url_mapping

#     try:
#         print("🔄 Reloading all face embeddings (max 100k)...")

#         photo_embeddings = {}
#         photo_url_mapping = {}
#         embedding_counter = 0
#         MAX_EMBEDDINGS = 100_000

#         for album in albums_collection.find({}).sort("last_updated", -1):  # 🔥 Sort by latest
            
#             for photo in album.get("photos", []):
#                 photo_id = photo.get("photo_id")
#                 photo_url = photo.get("image")
#                 embeddings = photo.get("face_embeddings", [])

#                 if not embeddings:
#                     continue

#                 if embedding_counter + len(embeddings) > MAX_EMBEDDINGS:
#                     print(f"⚠️ Stopping reload after {embedding_counter} embeddings loaded (limit reached).")
#                     break

#                 photo_url_mapping[photo_id] = photo_url

#                 face_list = []
#                 for face in embeddings:
#                     emb = np.array(face["embedding"])
#                     emb_norm = np.linalg.norm(emb)
#                     face_list.append((emb, emb_norm))
#                     embedding_counter += 1

#                 photo_embeddings[photo_id] = face_list

#             if embedding_counter >= MAX_EMBEDDINGS:
#                 break

#         print(f"✅ Reloaded {len(photo_embeddings)} photos and {embedding_counter} embeddings into RAM")

#         return {"message": "Embeddings reloaded successfully", "count": len(photo_embeddings)}

#     except Exception as e:
#         print("❌ Error reloading embeddings:", str(e))
#         return JSONResponse(content={"error": str(e)}, status_code=500)


# @app.post("/search-by-upload")
# async def search_by_upload(image: UploadFile = File(...)):
#     try:
#         # ✅ Acquire semaphore slot for concurrent uploads
#         await request_semaphore.acquire()
#     except asyncio.TimeoutError:
#         # ❌ Reject if no slot available
#         raise HTTPException(status_code=503, detail="Server busy. Try again later.")

#     if not image:
#         return JSONResponse(content={"error": "No image file provided"}, status_code=400)

#     if image.filename == "":
#         return JSONResponse(content={"error": "No file selected"}, status_code=400)

#     # ✅ STREAM upload, avoid full RAM loading
#     # image_obj = Image.open(image.file)
    
#     try:
#         # ✅ Convert uploaded file to PIL Image
#         image_obj = Image.open(image.file)

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

#         content_length = image.headers.get('content-length')
#         if content_length:
#             file_size_kb = int(content_length) / 1024
#             print(f"✅ Uploaded image file size: {file_size_kb:.2f} KB")
#         else:
#             file_size_kb = None
#             print("⚠️ Could not determine uploaded file size from headers")

#         if file_size_kb is not None:
#             print(f"✅ Uploaded image file size: {file_size_kb:.2f} KB")


#         print(f"✅ Uploaded image file size: {file_size_kb} KB")
#         print("✅ Uploaded image format:", image_obj.format, "| size:", image_obj.size, "| mode:", image_obj.mode)

#         if image_obj.mode == "RGBA":
#             image_obj = image_obj.convert("RGB")

#         # ✅ Extract faces from uploaded image
#         loop = asyncio.get_running_loop()
#         query_embeddings, extraction_error = await loop.run_in_executor(face_extract_executor, extract_faces, image_obj)


#         if extraction_error == "LowResolution":
#             return JSONResponse(content={"error": "Low resolution image. Please upload a photo with better resolution"}, status_code=400)

#         if extraction_error == "NoFaceDetected":
#             return JSONResponse(content={"error": "No face detected. Please upload a clear photo with a visible face."}, status_code=400)

#         if not query_embeddings:
#             return JSONResponse(content={"error": "Face extraction failed. Try again with another photo."}, status_code=400)

#         matched_photo_ids = set()

#         # ✅ Step 3: Compare with preloaded photo_embeddings instead of MongoDB
#         BLOCK_SIZE = 16000
#         photo_items = list(photo_embeddings.items())
#         total_blocks = (len(photo_items) + BLOCK_SIZE - 1) // BLOCK_SIZE

#         for block_idx in range(total_blocks):
#             start = block_idx * BLOCK_SIZE
#             end = min((block_idx + 1) * BLOCK_SIZE, len(photo_items))
#             block = photo_items[start:end]

#             for query_face in query_embeddings:
#                 query_emb = np.array(query_face["embedding"])
#                 query_norm = np.linalg.norm(query_emb)

#                 for photo_id, faces in block:
#                     for emb, emb_norm in faces:
#                         cosine_sim = np.dot(query_emb, emb) / (query_norm * emb_norm)
#                         if cosine_sim > 0.80:
#                             matched_photo_ids.add(photo_id)
#                             break

#             print(f"🔍 Searched block {block_idx + 1} ({start}-{end}), Total matches so far: {len(matched_photo_ids)}")

#             # Early stopping logic
#             if block_idx == 0 and len(matched_photo_ids) < 16:
#                 print(f"🛑 Less than 16 matches after first block. Stopping early.")
#                 break
#             if block_idx == 1 and len(matched_photo_ids) < 32:
#                 print(f"🛑 Less than 32 matches after second block. Stopping early.")
#                 break
#             # 🚫 No else: we just continue searching all blocks naturally!

#         if not matched_photo_ids:
#             return JSONResponse(
#                 content={"error": "No matching faces found in database. Try with another image."},
#                 status_code=404
#             )

#         matched_photos = [{"photo_id": pid, "image": photo_url_mapping[pid]} for pid in matched_photo_ids]

#         return {"photos": matched_photos}

#     except Exception as e:
#         print("❌ Error in /search-by-upload:", str(e))
#         return JSONResponse(content={"error": str(e)}, status_code=500)

#     finally:
#         request_semaphore.release()



# @app.post("/edit-album/{album_id}")
# async def edit_album(
#     album_id: str,
#     name: str = Form(None),
#     cover: UploadFile = File(None)
# ):
#     try:
#         update_fields = {}

#         # Find current album (so we can delete old cover if needed)
#         album = albums_collection.find_one({"_id": album_id})
#         if not album:
#             return JSONResponse(content={"error": "Album not found"}, status_code=404)

#         # If new name provided
#         if name:
#             update_fields["name"] = name

#         # If new cover provided
#         if cover:
#             image = Image.open(cover.file)
#             if image.mode == "RGBA":
#                 image = image.convert("RGB")
            
#             buffer = io.BytesIO()
#             image.save(buffer, format="JPEG", quality=50, optimize=True)
#             buffer.seek(0)
#             compressed_image = buffer.getvalue()

#             # Upload new cover to R2
#             cover_filename = f"covers/{uuid.uuid4().hex}.jpg"
#             cover_url = upload_to_r2(compressed_image, cover_filename)

#             # Delete old cover from R2
#             old_cover_url = album.get("cover")
#             if old_cover_url:
#                 delete_from_r2(old_cover_url)

#             # Update with new cover
#             update_fields["cover"] = cover_url

#         if not update_fields:
#             return JSONResponse(content={"error": "No updates provided"}, status_code=400)

#         albums_collection.update_one(
#             {"_id": album_id},
#             {"$set": update_fields}
#         )

#         return {"message": "Album updated successfully", "updates": update_fields}

#     except Exception as e:
#         print("❌ Album update error:", str(e))
#         return JSONResponse(content={"error": "Failed to update album"}, status_code=500)

from fastapi import FastAPI, Request, UploadFile, File, Form, Depends
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import requests
from fastapi import Query
from pydantic import BaseModel
from fastapi.responses import FileResponse
from typing import List
from PIL import ExifTags
import uuid
import os
import io
from io import BytesIO
from PIL import Image, ImageFilter
from rembg import new_session, remove
from deepface import DeepFace
import numpy as np
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from deepface.DeepFace import build_model
# from deepface.detectors import FaceDetector
import boto3
from botocore.client import Config
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi
import pymongo
from fastapi import Form
from fastapi import UploadFile, File
import asyncio
from fastapi import HTTPException
from concurrent.futures import ThreadPoolExecutor
import hashlib


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
print("🔧 Loading Facenet model...")
facenet_model = build_model("Facenet")
print("✅ Facenet model loaded.")

print("🔧 Loading u2netp model...")
rembg_session = new_session(model_name="u2netp")
print("✅ model loaded.")

print("🔧 Preloading all face embeddings grouped by photo_id (max 100k embeddings)...")

photo_embeddings = {}
photo_url_mapping = {}
embedding_counter = 0
MAX_EMBEDDINGS = 100_000

# Sort albums by last_updated descending (newest first)
for album in albums_collection.find({}).sort("last_updated", -1):
    for photo in album.get("photos", []):
        photo_id = photo.get("photo_id")
        photo_url = photo.get("image")
        embeddings = photo.get("face_embeddings", [])

        if not embeddings:
            continue

        # Check if adding these embeddings would exceed the limit
        if embedding_counter + len(embeddings) > MAX_EMBEDDINGS:
            print(f"⚠️ Stopping preload after {embedding_counter} embeddings loaded (limit reached).")
            break

        # Add photo URL mapping
        photo_url_mapping[photo_id] = photo_url

        # Add embeddings
        face_list = []
        for face in embeddings:
            emb = np.array(face.get("embedding"))
            emb_norm = np.linalg.norm(emb)  # ✅ precompute
            face_list.append((emb, emb_norm))  # ✅ save tuple
            embedding_counter += 1

        photo_embeddings[photo_id] = face_list

    # Double-break if limit is hit inside inner loop
    if embedding_counter >= MAX_EMBEDDINGS:
        break

print(f"✅ Preloaded {len(photo_embeddings)} photos and {embedding_counter} embeddings into RAM")

# FastAPI app
app = FastAPI()
MAX_UPLOAD_QUEUE = 22
upload_gallery_semaphore = asyncio.BoundedSemaphore(MAX_UPLOAD_QUEUE)
gallery_executor = ThreadPoolExecutor(max_workers=4) #
face_extract_executor = ThreadPoolExecutor(max_workers=2)
request_semaphore = asyncio.BoundedSemaphore(666)

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
    return {"message": "Backend is running successfully!!!!!"}


# ========== Upload Helpers ==========

def upload_to_r2(image_input, filename):
    try:
        ext = filename.split('.')[-1].lower()
        if ext in ["jpg", "jpeg"]:
            content_type = "image/jpeg"
        elif ext == "png":
            content_type = "image/png"
        else:
            content_type = "application/octet-stream"

        s3_client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=filename,
            Body=image_input,
            ContentType=content_type,
            ACL='public-read'
        )

        public_url = f"https://{PUBLIC_BUCKET_DOMAIN}/{filename}"
        return public_url
    except Exception as e:
        print("❌ Upload to R2 failed:", str(e))
        return None

def delete_from_r2(file_url):
    try:
        if not file_url:
            return
        key = file_url.split(PUBLIC_BUCKET_DOMAIN + "/")[-1]
        s3_client.delete_object(Bucket=R2_BUCKET_NAME, Key=key)
        print(f"✅ Deleted {key} from R2")
    except Exception as e:
        print(f"❌ Failed to delete {file_url}: {str(e)}")

def get_image_hash(image_bytes: bytes) -> str:
    """Return SHA256 hash of image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()

def is_duplicate_embedding(new_emb, existing_embs, threshold=0.95):
    """
    Check if new_emb is a near-duplicate of any existing embeddings.
    Uses cosine similarity, threshold defaults to 0.95 (very close).
    """
    new_emb = np.array(new_emb)
    new_norm = np.linalg.norm(new_emb)
    for emb, emb_norm in existing_embs:
        cosine_sim = np.dot(new_emb, emb) / (new_norm * emb_norm)
        if cosine_sim >= threshold:
            return True
    return False
# # ========== Face Extraction Helper ==========

def extract_faces(image_pil):
    temp_path = None  # Initialize temp_path early for safe deletion
    try:
        # Step 1: Save image temporarily
        temp_path = f"temp_{uuid.uuid4().hex}.jpg"
        image_pil.save(temp_path)

        print(f"🔍 Extracting faces using MTCNN from image: {temp_path}")

        # Step 2: Detect faces + Extract embeddings
        faces = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet",
            detector_backend="retinaface",  # 🔥 MTCNN Detector
            enforce_detection=True
        )

        if not faces:
            print("❌ No faces detected.")
            return [],  "NoFaceDetected"
        if len(faces) > 1:
            print("❌ Multiple faces detected. Rejecting image.")
            return [], "MultipleFacesDetected" 
          

        # Step 3: Filter faces with resolution >= 300px
        filtered_faces = []
        min_size = 80
        tolerance = 50

        for face in faces:
            area = face.get("facial_area", {})
            w, h = area.get("w", 0), area.get("h", 0)

            if w >= min_size and h >= min_size:
                # ✅ Both sides >= 200px
                filtered_faces.append(face)
            elif w < min_size and h < min_size:
                # ❌ Both sides < 200px
                print(f"⚠️ Skipping face {w}x{h} — both sides < {min_size}px")
                continue
            else:
                # One side >= 200px, check difference
                diff = abs(w - h)
                if diff <= tolerance:
                    filtered_faces.append(face)  # ✅ Accept
                else:
                    print(f"⚠️ Skipping face {w}x{h} — size difference {diff}px > {tolerance}px")

        if not filtered_faces:
            print("❌ No faces with required resolution.")
            return [], "LowResolution"

        # Step 4: Sort filtered faces by size (area w*h) and pick top 4
        faces_sorted = sorted(
            filtered_faces,
            key=lambda f: f.get("facial_area", {}).get("w", 0) * f.get("facial_area", {}).get("h", 0),
            reverse=True
        )
      # [:4]

        output_faces = []

        # Step 5: (Optional) Crop faces for preview/logging
        with Image.open(temp_path) as original:
            for face in faces_sorted:
                area = face.get("facial_area", {})
                x, y, w, h = area.get("x", 0), area.get("y", 0), area.get("w", 0), area.get("h", 0)

                cropped = original.crop((x, y, x + w, y + h))

                buffer = io.BytesIO()
                cropped.save(buffer, format="JPEG", quality=90)
                size_kb = len(buffer.getvalue()) / 1024

                print(f"🖼️ Face: {w}x{h} pixels | {round(size_kb, 2)} KB")

                output_faces.append({
                    "face_id": str(uuid.uuid4()),
                    "embedding": np.array(face["embedding"]).tolist(),  # ✅ Embedding already extracted
                    # (Optional: Could also save cropped image if needed)
                })

        print(f"✅ Returning {len(output_faces)} faces with size/resolution.")

        return output_faces, None

    except Exception as e:
        print("❌ Face extraction failed:", str(e))
        return [], "ExtractionError"

    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ========== Create Album ==========
@app.post("/create-album")
async def create_album(
    name: str = Form(...),
    date: str = Form(...),
    department: str = Form(""),
    districts: str = Form(""),
    with_cm: str = Form("without"),
    cover: UploadFile = File(...)
):
    try:
        image = Image.open(cover.file)
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break

            exif = image._getexif()
            if exif is not None:
                orientation_value = exif.get(orientation)
                if orientation_value == 3:
                    image = image.rotate(180, expand=True)
                elif orientation_value == 6:
                    image = image.rotate(270, expand=True)
                elif orientation_value == 8:
                    image = image.rotate(90, expand=True)
        except Exception as e:
            print(f"⚠️ EXIF rotation correction failed: {e}")

        if image.mode == "RGBA":
            image = image.convert("RGB")

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=50, optimize=True)
        buffer.seek(0)
        compressed_image = buffer.getvalue()

        cover_filename = f"covers/{uuid.uuid4().hex}.jpg"
        cover_url = upload_to_r2(compressed_image, cover_filename)
        now = datetime.utcnow()
        album = {
            "_id": str(uuid.uuid4()),
            "name": name,
            "date": date,
            "cover": cover_url,
            "department": department,
            "districts": [districts],
             "with_cm": with_cm,
            "photos": [],
            "last_updated": now 
        }

        albums_collection.insert_one(album)
        return {"message": "Album created successfully"}

    except Exception as e:
        print("❌ Album creation error:", str(e))
        return JSONResponse(content={"error": "Failed to process cover image"}, status_code=500)

def get_error_message(error):
    return {
        "NoFaceDetected": "No face detected",
        "LowResolution": "Face too small / low quality",
        "ExtractionError": "Face extraction failed",
        "DuplicateImage": "Duplicate image",
        "DuplicateEmbedding": "Same person already exists",
        "NoEmbeddings": "No face embeddings generated",
        "ProcessingError": "Image processing failed",
      "MultipleFacesDetected": "Image rejected: Multiple faces detected. Please upload single face selfie.",
    }.get(error, "Unknown error")


# # ========== Upload Photos to Gallery ==========

@app.post("/upload-gallery/{album_id}")
async def upload_gallery(album_id: str, photos: List[UploadFile] = File(...)):

    try:
        await upload_gallery_semaphore.acquire()
    except ValueError:
        raise HTTPException(status_code=503, detail="Server busy. Try again later.")

    try:
        if not photos:
            return JSONResponse(content={"error": "No files uploaded"}, status_code=400)

        new_photos = []
        rejected_files = []

        # ✅ MAIN LOOP
        for file in photos:
            try:
                image_bytes = await file.read()
                image_hash = get_image_hash(image_bytes)

                # ✅ Duplicate hash check
                duplicate = albums_collection.find_one({"photos.hash": image_hash})
                if duplicate:
                    rejected_files.append({
                        "file": file.filename,
                        "reason": get_error_message("DuplicateImage")
                    })
                    continue

                image = Image.open(io.BytesIO(image_bytes))

                if image.mode == "RGBA":
                    image = image.convert("RGB")

                # ✅ Face extraction
                loop = asyncio.get_running_loop()
                embeddings, extraction_error = await loop.run_in_executor(
                    gallery_executor, extract_faces, image
                )

                # if extraction_error in ["LowResolution", "NoFaceDetected"]:
                #     rejected_files.append({
                #         "file": file.filename,
                #         "reason": get_error_message(extraction_error)
                #     })
                #     continue

                # if extraction_error:
                #     print(f"⚠️ {file.filename} issue: {extraction_error}")
                if extraction_error:
                    rejected_files.append({
                        "file": file.filename,
                        "reason": get_error_message(extraction_error)
                   })
                   continue

                # if not embeddings:
                #     rejected_files.append({
                #         "file": file.filename,
                #         "reason": get_error_message("NoEmbeddings")
                #     })
                #     continue
                if not embeddings:
                    embeddings = []

                # ✅ Duplicate embedding check
                # is_duplicate = False
                # for emb in embeddings:
                #     for faces in photo_embeddings.values():
                #         if is_duplicate_embedding(emb["embedding"], faces):
                #             is_duplicate = True
                #             break
                #     if is_duplicate:
                #         break

                # if is_duplicate:
                #     rejected_files.append({
                #         "file": file.filename,
                #         "reason": get_error_message("DuplicateEmbedding")
                #     })
                #     continue

                # ✅ Compress & Upload
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=40, optimize=True)
                buffer.seek(0)

                filename = f"gallery/{uuid.uuid4().hex}.jpg"
                image_url = upload_to_r2(buffer.getvalue(), filename)

                photo = {
                    "photo_id": str(uuid.uuid4()),
                    "image": image_url,
                    "hash": image_hash,
                    "face_embeddings": embeddings
                }

                new_photos.append(photo)

            except Exception as e:
                print(f"❌ Failed to process {file.filename}: {e}")

                try:
                    buffer = io.BytesIO(image_bytes)
                    filename = f"gallery/{uuid.uuid4().hex}.jpg"
                    image_url = upload_to_r2(buffer.getvalue(), filename)

                    photo = {
                        "photo_id": str(uuid.uuid4()),
                        "image": image_url,
                        "hash": image_hash,
                        "face_embeddings": []
                    }

                    new_photos.append(photo)

                except:
                    rejected_files.append({
                        "file": file.filename,
                        "reason": get_error_message("ProcessingError")
                    })

        # ✅ AFTER LOOP (IMPORTANT FIX)
        if new_photos:
            now = datetime.utcnow()
            albums_collection.update_one(
                {"_id": album_id},
                {
                    "$push": {"photos": {"$each": new_photos}},
                    "$set": {"last_updated": now}
                }
            )

        return JSONResponse(content={
            "message": "Upload complete",
            "uploaded": len(new_photos),
            "rejected": rejected_files
        }, status_code=201)

    except Exception as e:
        print("❌ Upload failed:", str(e))
        return JSONResponse(content={"error": "Upload failed"}, status_code=500)

    finally:
        upload_gallery_semaphore.release()

# @app.post("/upload-gallery/{album_id}")
# async def upload_gallery(album_id: str, photos: List[UploadFile] = File(...)):
#     try:
#         await upload_gallery_semaphore.acquire()
#     except ValueError:
#         raise HTTPException(status_code=503, detail="Server busy. Try again later.")

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

#                 image_hash = get_image_hash(image_bytes)

#                 duplicate = albums_collection.find_one({"photos.hash": image_hash})
#                 if duplicate:
#                     print(f"⚠️ Duplicate image detected (hash match): {file.filename}")
#                     # rejected_files.append(file.filename)
#                  rejected_files.append({
#     "file": file.filename,
#     "reason": get_error_message("DuplicateImage")
# })
#                     continue


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
#                 loop = asyncio.get_running_loop()
#                 embeddings, extraction_error = await loop.run_in_executor(gallery_executor, extract_faces, image)

#                 if extraction_error == "LowResolution" or extraction_error == "NoFaceDetected":
#                     print(f"❌ Face extraction failed for {file.filename} due to {extraction_error}")
#                     # rejected_files.append(file.filename)
#                 rejected_files.append({
#     "file": file.filename,
#     "reason": get_error_message(extraction_error)
# })
#                     continue

#                 if not embeddings:
#                     print(f"❌ No embeddings generated for {file.filename}")
#                     # rejected_files.append(file.filename)
#                       rejected_files.append({
#     "file": file.filename,
#     "reason": get_error_message("NoEmbeddings")
# })
#                     continue

#                                 # ✅ Embedding-level duplicate check
#                 is_duplicate = False
#                 for emb in embeddings:
#                     for faces in photo_embeddings.values():
#                         if is_duplicate_embedding(emb["embedding"], faces):
#                             is_duplicate = True
#                             break
#                     if is_duplicate:
#                         break

#                 if is_duplicate:
#                     print(f"⚠️ Duplicate embedding detected for {file.filename}")
#                     # rejected_files.append(file.filename)
#   rejected_files.append({
#     "file": file.filename,
#     "reason": get_error_message("DuplicateImage")
# })
                  
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
#                     "hash": image_hash,   # save hash in DB
#                     "face_embeddings": embeddings
#                 }
#                 new_photos.append(photo)

#             except Exception as e:
#                 print(f"❌ Failed to process {file.filename}: {e}")
#                 rejected_files.append(file.filename)for file in photos:
    # try:
    #     image_bytes = await file.read()

    #     image_hash = get_image_hash(image_bytes)

    #     # ✅ DUPLICATE HASH CHECK
    #     duplicate = albums_collection.find_one({"photos.hash": image_hash})
    #     if duplicate:
    #         rejected_files.append({
    #             "file": file.filename,
    #             "reason": get_error_message("DuplicateImage")
    #         })
    #         continue

    #     image = Image.open(io.BytesIO(image_bytes))

    #     if image.mode == "RGBA":
    #         image = image.convert("RGB")

    #     # ✅ FACE EXTRACTION
    #     loop = asyncio.get_running_loop()
    #     embeddings, extraction_error = await loop.run_in_executor(
    #         gallery_executor, extract_faces, image
    #     )

    #     if extraction_error in ["LowResolution", "NoFaceDetected"]:
    #         rejected_files.append({
    #             "file": file.filename,
    #             "reason": get_error_message(extraction_error)
    #         })
    #         continue

    #     if not embeddings:
    #         rejected_files.append({
    #             "file": file.filename,
    #             "reason": get_error_message("NoEmbeddings")
    #         })
    #         continue

    #     # ✅ DUPLICATE EMBEDDING CHECK
    #     is_duplicate = False
    #     for emb in embeddings:
    #         for faces in photo_embeddings.values():
    #             if is_duplicate_embedding(emb["embedding"], faces):
    #                 is_duplicate = True
    #                 break
    #         if is_duplicate:
    #             break

    #     if is_duplicate:
    #         rejected_files.append({
    #             "file": file.filename,
    #             "reason": get_error_message("DuplicateEmbedding")
    #         })
    #         continue

    #     # ✅ SAVE IMAGE
    #     buffer = io.BytesIO()
    #     image.save(buffer, format="JPEG", quality=40, optimize=True)
    #     buffer.seek(0)

    #     filename = f"gallery/{uuid.uuid4().hex}.jpg"
    #     image_url = upload_to_r2(buffer.getvalue(), filename)

    #     photo = {
    #         "photo_id": str(uuid.uuid4()),
    #         "image": image_url,
    #         "hash": image_hash,
    #         "face_embeddings": embeddings
    #     }

    #     new_photos.append(photo)

    # except Exception as e:
    #     print(f"❌ Failed to process {file.filename}: {e}")
    #     rejected_files.append({
    #         "file": file.filename,
    #         "reason": get_error_message("ProcessingError")
    #     })
      

    #     # ✅ Update album in database
    #     if new_photos:
    #         now = datetime.utcnow()
    #         albums_collection.update_one(
    #             {"_id": album_id},
    #             {"$push": {"photos": {"$each": new_photos}},
    #              "$set": {"last_updated": now} 
    #             }
    #         )

    #     return JSONResponse(content={
    #         "message": "Upload complete",
    #         "uploaded": len(new_photos),
    #         "rejected": rejected_files
    #     }, status_code=201)

    # except Exception as e:
    #     print("❌ Upload failed:", str(e))
    #     return JSONResponse(content={"error": "Upload failed"}, status_code=500)
    
    # finally:
    #     upload_gallery_semaphore.release()

@app.post("/reload-embeddings")
async def reload_embeddings():
    global photo_embeddings, photo_url_mapping

    try:
        print("🔄 Reloading all face embeddings (max 100k)...")

        photo_embeddings = {}
        photo_url_mapping = {}
        embedding_counter = 0
        MAX_EMBEDDINGS = 100_000

        for album in albums_collection.find({}).sort("last_updated", -1):  # 🔥 Sort by latest
            
            for photo in album.get("photos", []):
                photo_id = photo.get("photo_id")
                photo_url = photo.get("image")
                embeddings = photo.get("face_embeddings", [])

                if not embeddings:
                    continue

                if embedding_counter + len(embeddings) > MAX_EMBEDDINGS:
                    print(f"⚠️ Stopping reload after {embedding_counter} embeddings loaded (limit reached).")
                    break

                photo_url_mapping[photo_id] = photo_url

                face_list = []
                for face in embeddings:
                    emb = np.array(face["embedding"])
                    emb_norm = np.linalg.norm(emb)
                    face_list.append((emb, emb_norm))
                    embedding_counter += 1

                photo_embeddings[photo_id] = face_list

            if embedding_counter >= MAX_EMBEDDINGS:
                break

        print(f"✅ Reloaded {len(photo_embeddings)} photos and {embedding_counter} embeddings into RAM")

        return {"message": "Embeddings reloaded successfully", "count": len(photo_embeddings)}

    except Exception as e:
        print("❌ Error reloading embeddings:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/search-by-upload")
async def search_by_upload(image: UploadFile = File(...)):
    try:
        # ✅ Acquire semaphore slot for concurrent uploads
        await request_semaphore.acquire()
    except asyncio.TimeoutError:
        # ❌ Reject if no slot available
        raise HTTPException(status_code=503, detail="Server busy. Try again later.")

    if not image:
        return JSONResponse(content={"error": "No image file provided"}, status_code=400)

    if image.filename == "":
        return JSONResponse(content={"error": "No file selected"}, status_code=400)

    # ✅ STREAM upload, avoid full RAM loading
    # image_obj = Image.open(image.file)
    
    try:
        # ✅ Convert uploaded file to PIL Image
        image_obj = Image.open(image.file)

        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break

            exif = image_obj._getexif()
            if exif is not None:
                orientation_value = exif.get(orientation)
                if orientation_value == 3:
                    image_obj = image_obj.rotate(180, expand=True)
                elif orientation_value == 6:
                    image_obj = image_obj.rotate(270, expand=True)
                elif orientation_value == 8:
                    image_obj = image_obj.rotate(90, expand=True)

        except Exception as e:
            print(f"⚠️ EXIF rotation correction failed: {e}")

        content_length = image.headers.get('content-length')
        if content_length:
            file_size_kb = int(content_length) / 1024
            print(f"✅ Uploaded image file size: {file_size_kb:.2f} KB")
        else:
            file_size_kb = None
            print("⚠️ Could not determine uploaded file size from headers")

        if file_size_kb is not None:
            print(f"✅ Uploaded image file size: {file_size_kb:.2f} KB")


        print(f"✅ Uploaded image file size: {file_size_kb} KB")
        print("✅ Uploaded image format:", image_obj.format, "| size:", image_obj.size, "| mode:", image_obj.mode)

        if image_obj.mode == "RGBA":
            image_obj = image_obj.convert("RGB")

        # ✅ Extract faces from uploaded image
        loop = asyncio.get_running_loop()
        query_embeddings, extraction_error = await loop.run_in_executor(face_extract_executor, extract_faces, image_obj)


        if extraction_error == "LowResolution":
            return JSONResponse(content={"error": "Low resolution image. Please upload a photo with better resolution"}, status_code=400)

        if extraction_error == "NoFaceDetected":
            return JSONResponse(content={"error": "No face detected. Please upload a clear photo with a visible face."}, status_code=400)

        if not query_embeddings:
            return JSONResponse(content={"error": "Face extraction failed. Try again with another photo."}, status_code=400)

        matched_photo_ids = set()

        # ✅ Step 3: Compare with preloaded photo_embeddings instead of MongoDB
        BLOCK_SIZE = 16000
        photo_items = list(photo_embeddings.items())
        total_blocks = (len(photo_items) + BLOCK_SIZE - 1) // BLOCK_SIZE

        for block_idx in range(total_blocks):
            start = block_idx * BLOCK_SIZE
            end = min((block_idx + 1) * BLOCK_SIZE, len(photo_items))
            block = photo_items[start:end]

            for query_face in query_embeddings:
                query_emb = np.array(query_face["embedding"])
                query_norm = np.linalg.norm(query_emb)

                for photo_id, faces in block:
                    for emb, emb_norm in faces:
                        cosine_sim = np.dot(query_emb, emb) / (query_norm * emb_norm)
                        if cosine_sim > 0.75:
                            matched_photo_ids.add(photo_id)
                            break

            print(f"🔍 Searched block {block_idx + 1} ({start}-{end}), Total matches so far: {len(matched_photo_ids)}")

            # Early stopping logic
            # if block_idx == 0 and len(matched_photo_ids) < 16:
            #     print(f"🛑 Less than 16 matches after first block. Stopping early.")
            #     break
            # if block_idx == 1 and len(matched_photo_ids) < 32:
            #     print(f"🛑 Less than 32 matches after second block. Stopping early.")
            #     break
            # 🚫 No else: we just continue searching all blocks naturally!

        if not matched_photo_ids:
            return JSONResponse(
                content={"error": "No matching faces found in database. Try with another image."},
                status_code=404
            )

        matched_photos = [{"photo_id": pid, "image": photo_url_mapping[pid]} for pid in matched_photo_ids]

        return {"photos": matched_photos}

    except Exception as e:
        print("❌ Error in /search-by-upload:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        request_semaphore.release()



@app.post("/edit-album/{album_id}")
async def edit_album(
    album_id: str,
    name: str = Form(None),
    cover: UploadFile = File(None)
):
    try:
        update_fields = {}

        # Find current album (so we can delete old cover if needed)
        album = albums_collection.find_one({"_id": album_id})
        if not album:
            return JSONResponse(content={"error": "Album not found"}, status_code=404)

        # If new name provided
        if name:
            update_fields["name"] = name

        # If new cover provided
        if cover:
            image = Image.open(cover.file)
            if image.mode == "RGBA":
                image = image.convert("RGB")
            
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=50, optimize=True)
            buffer.seek(0)
            compressed_image = buffer.getvalue()

            # Upload new cover to R2
            cover_filename = f"covers/{uuid.uuid4().hex}.jpg"
            cover_url = upload_to_r2(compressed_image, cover_filename)

            # Delete old cover from R2
            old_cover_url = album.get("cover")
            if old_cover_url:
                delete_from_r2(old_cover_url)

            # Update with new cover
            update_fields["cover"] = cover_url

        if not update_fields:
            return JSONResponse(content={"error": "No updates provided"}, status_code=400)

        albums_collection.update_one(
            {"_id": album_id},
            {"$set": update_fields}
        )

        return {"message": "Album updated successfully", "updates": update_fields}

    except Exception as e:
        print("❌ Album update error:", str(e))
        return JSONResponse(content={"error": "Failed to update album"}, status_code=500)

@app.get("/grouped-faces")
async def grouped_faces():
    try:
        threshold = 0.75  # same as your search logic

        clusters = []

        for photo_id, faces in photo_embeddings.items():
            photo_url = photo_url_mapping.get(photo_id)

            for emb, emb_norm in faces:
                matched_cluster = None

                # 🔍 Try to match existing cluster
                for cluster in clusters:
                    rep_emb, rep_norm = cluster["rep_embedding"]

                    cosine_sim = np.dot(emb, rep_emb) / (emb_norm * rep_norm)

                    if cosine_sim > threshold:
                        matched_cluster = cluster
                        break

                if matched_cluster:
                    matched_cluster["count"] += 1
                    matched_cluster["photos"].add(photo_url)
                else:
                    clusters.append({
                        "rep_embedding": (emb, emb_norm),
                        "count": 1,
                        "photos": set([photo_url])
                    })

        # 🔄 Convert to response
        result = []
        for idx, cluster in enumerate(clusters):
            result.append({
                "person_id": str(idx),
                "count": cluster["count"],
                "thumbnail": list(cluster["photos"])[0],
                "photos": list(cluster["photos"])
            })

        return {"people": result}

    except Exception as e:
        print("❌ Error in grouped-faces:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

