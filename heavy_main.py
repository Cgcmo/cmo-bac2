# # main.py

# from fastapi import FastAPI, Request, UploadFile, File, Form
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from pydantic import BaseModel
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
# from datetime import datetime
# from deepface.DeepFace import build_model
# import boto3
# from botocore.client import Config
# import requests
# from dotenv import load_dotenv
# from pymongo import MongoClient
# import certifi
# from huey import MemoryHuey


# # Load .env
# load_dotenv()

# general_huey = MemoryHuey(name="general")
# search_huey = MemoryHuey(name="search")

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

# # MongoDB
# client = MongoClient("mongodb://localhost:27017/")

# photo_gallery_db = client["photo_gallery"]
# albums_collection = photo_gallery_db["albums"]

# dist_and_depart_db = client["dist_and_depart"]
# districts_collection = dist_and_depart_db["districts"]
# departments_collection = dist_and_depart_db["departments"]
# banners_collection = dist_and_depart_db["banners"]

# auth_db = client["auth_db"]
# users_collection = auth_db["users"]
# clients_collection = auth_db["clients"]

# # Load Facenet model once
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

# # FastAPI app
# app = FastAPI()

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
#     return {"message": "Backend is running successfully!"}


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


# # ========== Face Extraction Helper ==========
# def extract_faces(image_pil):
#     temp_path = None
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

#         print(f"🔍 Extracting faces from: {temp_path}")

#         # ✅ Step 4: Use DeepFace for face detection and embedding
#         faces = DeepFace.represent(
#             img_path=temp_path,
#             model_name="Facenet",
#             detector_backend="mtcnn",
#             enforce_detection=True
#         )

#         if not faces:
#             print("❌ No faces detected.")
#             return []

#         # ✅ Sort by size and keep top 4
#         faces_sorted = sorted(
#             faces,
#             key=lambda f: f.get("facial_area", {}).get("w", 0) * f.get("facial_area", {}).get("h", 0),
#             reverse=True
#         )[:4]


#         output_faces = []
#         with Image.open(temp_path) as original:
#                     for face in faces_sorted:
#                         area = face.get("facial_area", {})
#                         x, y, w, h = area.get("x", 0), area.get("y", 0), area.get("w", 0), area.get("h", 0)

#                         # ✅ Skip if both width and height are below 450
#                         if w < 450 and h < 450:
#                             print(f"⚠️  Skipping small face: {w}x{h} resolution")
#                             continue

#                         cropped = original.crop((x, y, x + w, y + h))

#                         buffer = io.BytesIO()
#                         cropped.save(buffer, format="JPEG", quality=90)
#                         size_kb = len(buffer.getvalue()) / 1024

#                         print(f"🖼️  Face: {w}x{h} pixels | {round(size_kb, 2)} KB")

#                         output_faces.append({
#                             "face_id": str(uuid.uuid4()),
#                             "embedding": np.array(face["embedding"]).tolist(),
#                             "resolution": f"{w}x{h}",
#                             "size_kb": round(size_kb, 2)
#                         })

#         # ✅ Now temp file can be deleted safely
#         if temp_path and os.path.exists(temp_path):
#             os.remove(temp_path)

#             print(f"✅ Returning {len(output_faces)} faces with size/resolution")

#             return output_faces

#     except Exception as e:
#         print("❌ Face extraction failed:", str(e))
#         if temp_path and os.path.exists(temp_path):
#             os.remove(temp_path)
#         return []


# # ========== Create Album ==========
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


# class AlbumDeleteRequest(BaseModel):
#     albumIds: list[str]

# @app.delete("/delete-albums")
# async def delete_multiple_albums(data: AlbumDeleteRequest):
#     album_ids = data.albumIds

#     if not album_ids:
#         return JSONResponse(content={"error": "No album IDs provided"}, status_code=400)

#     try:
#         for album_id in album_ids:
#             album = albums_collection.find_one({"_id": album_id})
#             if album:
#                 delete_from_r2(album.get("cover"))
#                 for photo in album.get("photos", []):
#                     delete_from_r2(photo.get("image"))

#         result = albums_collection.delete_many({"_id": {"$in": album_ids}})
#         return {"message": f"Deleted {result.deleted_count} albums successfully"}

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)



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




# main.py

# from fastapi import FastAPI, Request, UploadFile, File, Form
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from pydantic import BaseModel
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
# from datetime import datetime
# from deepface.DeepFace import build_model
# import boto3
# from botocore.client import Config
# import requests
# from dotenv import load_dotenv
# from pymongo import MongoClient
# import certifi
# from huey import RedisHuey
# from huey.api import Result


# # Load .env
# load_dotenv()

# general_huey = RedisHuey('general')
# search_huey = RedisHuey('search') 

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

# # MongoDB
# client = MongoClient("mongodb://localhost:27017/")

# photo_gallery_db = client["photo_gallery"]
# albums_collection = photo_gallery_db["albums"]

# dist_and_depart_db = client["dist_and_depart"]
# districts_collection = dist_and_depart_db["districts"]
# departments_collection = dist_and_depart_db["departments"]
# banners_collection = dist_and_depart_db["banners"]

# auth_db = client["auth_db"]
# users_collection = auth_db["users"]
# clients_collection = auth_db["clients"]

# # Load Facenet model once
# print("🔧 Loading Facenet model...")
# facenet_model = build_model("Facenet")
# print("✅ Facenet model loaded.")

# print("🔧 Loading u2netp model...")
# rembg_session = new_session(model_name="u2netp")
# print("✅ model loaded.")


# def preload_embeddings(limit=100000):
#     global photo_embeddings, photo_url_mapping

#     print("🔄 Preloading face embeddings... (latest first, max", limit, ")")

#     photo_embeddings = {}
#     photo_url_mapping = {}

#     total_loaded = 0

#     albums_cursor = albums_collection.find({}).sort("last_updated", -1)

#     for album in albums_cursor:
#         for photo in album.get("photos", []):
#             photo_id = photo.get("photo_id")
#             photo_url = photo.get("image")
#             embeddings = photo.get("face_embeddings", [])

#             if not embeddings:
#                 continue

#             face_list = []
#             for face in embeddings:
#                 if total_loaded >= limit:
#                     break
#                 emb = np.array(face.get("embedding"))
#                 emb_norm = np.linalg.norm(emb)
#                 face_list.append((emb, emb_norm))
#                 total_loaded += 1

#             if face_list:
#                 photo_embeddings[photo_id] = face_list
#                 photo_url_mapping[photo_id] = photo_url

#             if total_loaded >= limit:
#                 break
#         if total_loaded >= limit:
#             break

#     print(f"✅ Preloaded {total_loaded} face embeddings from {len(photo_embeddings)} photos.")
#     return total_loaded

# # ✅ Preload embeddings (latest first, max 100k) at startup
# preload_embeddings()

# # FastAPI app
# app = FastAPI()

# task_status = {
#     "search": {"running": False, "total_executed": 0},
#     "general": {"running": False, "total_executed": 0}
# }

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
#     return {"message": "Backend is running successfully!"}

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


# # ========== Face Extraction Helper ==========
# def extract_faces(image_pil):
#     temp_path = None
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

#         print(f"🔍 Extracting faces from: {temp_path}")

#         # ✅ Step 4: Use DeepFace for face detection and embedding
#         faces = DeepFace.represent(
#             img_path=temp_path,
#             model_name="Facenet",
#             detector_backend="mtcnn",
#             enforce_detection=True
#         )

#         if not faces:
#             print("❌ No faces detected.")
#             return []

#         # ✅ Sort by size and keep top 4
#         faces_sorted = sorted(
#             faces,
#             key=lambda f: f.get("facial_area", {}).get("w", 0) * f.get("facial_area", {}).get("h", 0),
#             reverse=True
#         )[:4]


#         output_faces = []
#         with Image.open(temp_path) as original:
#                     for face in faces_sorted:
#                         area = face.get("facial_area", {})
#                         x, y, w, h = area.get("x", 0), area.get("y", 0), area.get("w", 0), area.get("h", 0)

#                         # ✅ Skip if both width and height are below 450
#                         if w < 450 and h < 450:
#                             print(f"⚠️  Skipping small face: {w}x{h} resolution")
#                             continue

#                         cropped = original.crop((x, y, x + w, y + h))

#                         buffer = io.BytesIO()
#                         cropped.save(buffer, format="JPEG", quality=90)
#                         size_kb = len(buffer.getvalue()) / 1024

#                         print(f"🖼️  Face: {w}x{h} pixels | {round(size_kb, 2)} KB")

#                         output_faces.append({
#                             "face_id": str(uuid.uuid4()),
#                             "embedding": np.array(face["embedding"]).tolist(),
#                             "resolution": f"{w}x{h}",
#                             "size_kb": round(size_kb, 2)
#                         })

#         # ✅ Now temp file can be deleted safely
#         if temp_path and os.path.exists(temp_path):
#             os.remove(temp_path)

#             print(f"✅ Returning {len(output_faces)} faces with size/resolution")

#             return output_faces

#     except Exception as e:
#         print("❌ Face extraction failed:", str(e))
#         if temp_path and os.path.exists(temp_path):
#             os.remove(temp_path)
#         return []


# # ========== Create Album ==========
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
#             "photos": [],
#             "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
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
#             update_result = albums_collection.update_one(
#                 {"_id": album_id},
#                 {
#                     "$set": {"last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")},
#                     "$push": {"photos": {"$each": new_photos}}
#                 }
#             )
#             print("Matched:", update_result.matched_count, "| Modified:", update_result.modified_count)



#         return JSONResponse(content={
#             "message": "Upload complete",
#             "uploaded": len(new_photos),
#             "rejected": rejected_files
#         }, status_code=201)

#     except Exception as e:
#         print("❌ Upload failed:", str(e))
#         return JSONResponse(content={"error": "Upload failed"}, status_code=500)



# @app.post("/reload-embeddings")
# async def reload_embeddings():
#     try:
#         total = preload_embeddings()
#         return {"message": "Embeddings reloaded", "embeddings_loaded": total}
#     except Exception as e:
#         print("❌ Error during embedding preload:", str(e))
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# @search_huey.task()  # removed bind=True
# def process_search_by_upload(img_bytes, submitted_time):
#     from PIL import Image
#     import numpy as np
#     import io
#     from datetime import datetime


#     # ✅ Expiry check
#     now = datetime.utcnow()
#     submitted = datetime.strptime(submitted_time, "%Y-%m-%dT%H:%M:%S")
#     diff = (now - submitted).total_seconds()
#     if diff > 180:
#         print("❌ Task expired, skipping.")
#         return {"error": "Task expired (180s limit exceeded)"}
        
#     image_obj = Image.open(io.BytesIO(img_bytes))
#     try:
#         for orientation in ExifTags.TAGS.keys():
#             if ExifTags.TAGS[orientation] == 'Orientation':
#                 break
#         exif = image_obj._getexif()
#         if exif is not None:
#             orientation_value = exif.get(orientation)
#             if orientation_value == 3:
#                 image_obj = image_obj.rotate(180, expand=True)
#             elif orientation_value == 6:
#                 image_obj = image_obj.rotate(270, expand=True)
#             elif orientation_value == 8:
#                 image_obj = image_obj.rotate(90, expand=True)
#     except Exception as e:
#         print(f"⚠️ EXIF rotation correction failed: {e}")

#     image_obj.load()  # force decoding
#     file_size_kb = len(img_bytes) / 1024  # in KB
#     file_size_kb = round(file_size_kb, 2)  # rounded to 2 decimals

#     print(f"✅ Uploaded image file size: {file_size_kb} KB")
#     print("✅ Uploaded image format:", image_obj.format, "| size:", image_obj.size, "| mode:", image_obj.mode)


#     if image_obj.mode == "RGBA":
#         image_obj = image_obj.convert("RGB")

#     query_embeddings = extract_faces(image_obj)
#     if not query_embeddings:
#         return {"error": "No face found in uploaded photo"}

#     matched_photo_ids = set()
#     for query_face in query_embeddings:
#         query_emb = np.array(query_face["embedding"])
#         query_norm = np.linalg.norm(query_emb)
#         for photo_id, faces in photo_embeddings.items():
#             for emb, emb_norm in faces:
#                 cosine_sim = np.dot(query_emb, emb) / (query_norm * emb_norm)
#                 if cosine_sim > 0.75:
#                     matched_photo_ids.add(photo_id)
#                     break

#     if not matched_photo_ids:
#             return JSONResponse(
#                 content={"error": "No matching faces found in database, either there is no photo of uploaded face or face is not clear"},
#                 status_code=404
#             )

#     matched_photos = [{"photo_id": pid, "image": photo_url_mapping[pid]} for pid in matched_photo_ids]
#     return {"photos": matched_photos}

# @app.post("/search-by-upload")
# async def search_by_upload(image: UploadFile = File(...)):
#     if not image or image.filename == "":
#         return JSONResponse(content={"error": "No image file provided"}, status_code=400)

#     img_bytes = await image.read()
#     submitted_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

#     # ✅ Correct call with `args=...` and `delay=0`
#     task = process_search_by_upload.schedule(args=(img_bytes, submitted_time), delay=0)

#     return JSONResponse(content={"task_id": task.id})

# @app.get("/debug/preload-status")
# def preload_status():
#     from datetime import datetime

#     # Optional: Store last preload time globally when preload_embeddings() is called
#     global last_preload_time
#     return {
#         "total_preloaded_photos": len(photo_embeddings),
#         "sample_photo_ids": list(photo_embeddings.keys())[:5],
#         "sample_image_urls": [photo_url_mapping[pid] for pid in list(photo_url_mapping.keys())[:5]],
#         "last_preload_time": last_preload_time if 'last_preload_time' in globals() else "Unknown"
#     }

# @app.get("/task-status/{task_id}")
# def get_task_status(task_id: str):
#     try:
#         result = search_huey.result(task_id)  # ✅ Already returns final result (dict or None)

#         if result is None:
#             return {"status": "pending"}  # task not yet done

#         return {"status": "done", "result": result}

#     except Exception as e:
#         return {"status": "error", "error": str(e)}

    
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


# class AlbumDeleteRequest(BaseModel):
#     albumIds: list[str]

# @app.delete("/delete-albums")
# async def delete_multiple_albums(data: AlbumDeleteRequest):
#     album_ids = data.albumIds

#     if not album_ids:
#         return JSONResponse(content={"error": "No album IDs provided"}, status_code=400)

#     try:
#         for album_id in album_ids:
#             album = albums_collection.find_one({"_id": album_id})
#             if album:
#                 delete_from_r2(album.get("cover"))
#                 for photo in album.get("photos", []):
#                     delete_from_r2(photo.get("image"))

#         result = albums_collection.delete_many({"_id": {"$in": album_ids}})
#         return {"message": f"Deleted {result.deleted_count} albums successfully"}

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)



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







from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic import BaseModel
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
from datetime import datetime
from deepface.DeepFace import build_model
import boto3
from botocore.client import Config
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi
from huey import RedisHuey
from huey.api import Result
import redis
from embedding_store import preload_embeddings, photo_embeddings, photo_url_mapping
import pymongo


# Load .env
load_dotenv()

general_huey = RedisHuey('general')
search_huey = RedisHuey('search') 

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

# MongoDB
client = pymongo.MongoClient(
    os.getenv("MONGO_DB_URL"),
    tls=True,
    tlsCAFile=certifi.where()
)
redis_client = redis.Redis(host='localhost', port=6379, db=0)


photo_gallery_db = client["photo_gallery"]
albums_collection = photo_gallery_db["albums"]

dist_and_depart_db = client["dist_and_depart"]
districts_collection = dist_and_depart_db["districts"]
departments_collection = dist_and_depart_db["departments"]
banners_collection = dist_and_depart_db["banners"]

auth_db = client["auth_db"]
users_collection = auth_db["users"]
clients_collection = auth_db["clients"]

# Load Facenet model once
print("🔧 Loading Facenet model...")
facenet_model = build_model("Facenet")
print("✅ Facenet model loaded.")

print("🔧 Loading u2netp model...")
rembg_session = new_session(model_name="u2netp")
print("✅ model loaded.")

preload_embeddings()

# FastAPI app
app = FastAPI()

task_status = {
    "search": {"running": False, "total_executed": 0},
    "general": {"running": False, "total_executed": 0}
}

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


# ========== Face Extraction Helper ==========
def extract_faces(image_pil):
    temp_path = None
    try:
        # ✅ Step 1: Remove background
        buffered = io.BytesIO()
        image_pil.save(buffered, format="PNG")
        input_bytes = buffered.getvalue()
        output_bytes = remove(input_bytes, session=rembg_session)
        image_no_bg = Image.open(io.BytesIO(output_bytes)).convert("RGB")

        # ✅ Step 2: Sharpen the image
        sharpened = image_no_bg.filter(ImageFilter.SHARPEN)

        # ✅ Step 3: Save temporarily for DeepFace
        temp_path = f"temp_{uuid.uuid4().hex}.jpg"
        sharpened.save(temp_path)

        print(f"🔍 Extracting faces from: {temp_path}")

        # ✅ Step 4: Use DeepFace for face detection and embedding
        faces = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet",
            detector_backend="mtcnn",
            enforce_detection=True
        )

        if not faces:
            print("❌ No faces detected.")
            return []

        # ✅ Sort by size and keep top 4
        faces_sorted = sorted(
            faces,
            key=lambda f: f.get("facial_area", {}).get("w", 0) * f.get("facial_area", {}).get("h", 0),
            reverse=True
        )[:4]


        output_faces = []
        with Image.open(temp_path) as original:
                    for face in faces_sorted:
                        area = face.get("facial_area", {})
                        x, y, w, h = area.get("x", 0), area.get("y", 0), area.get("w", 0), area.get("h", 0)

                        # ✅ Skip if both width and height are below 450
                        if w < 450 and h < 450:
                            print(f"⚠️  Skipping small face: {w}x{h} resolution")
                            continue

                        cropped = original.crop((x, y, x + w, y + h))

                        buffer = io.BytesIO()
                        cropped.save(buffer, format="JPEG", quality=90)
                        size_kb = len(buffer.getvalue()) / 1024

                        print(f"🖼️  Face: {w}x{h} pixels | {round(size_kb, 2)} KB")

                        output_faces.append({
                            "face_id": str(uuid.uuid4()),
                            "embedding": np.array(face["embedding"]).tolist(),
                            
                        })

        # ✅ Now temp file can be deleted safely
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

            print(f"✅ Returning {len(output_faces)} faces with size/resolution")

            return output_faces

    except Exception as e:
        print("❌ Face extraction failed:", str(e))
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return []


# ========== Create Album ==========
@app.post("/create-album")
async def create_album(
    name: str = Form(...),
    date: str = Form(...),
    department: str = Form(""),
    districts: str = Form(""),
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

        album = {
            "_id": str(uuid.uuid4()),
            "name": name,
            "date": date,
            "cover": cover_url,
            "department": department,
            "districts": [districts],
            "photos": [],
            "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        }

        albums_collection.insert_one(album)
        return {"message": "Album created successfully"}

    except Exception as e:
        print("❌ Album creation error:", str(e))
        return JSONResponse(content={"error": "Failed to process cover image"}, status_code=500)


# ========== Upload Photos to Gallery ==========

@app.post("/upload-gallery/{album_id}")
async def upload_gallery(album_id: str, photos: List[UploadFile] = File(...)):
    try:
        if not photos:
            return JSONResponse(content={"error": "No files uploaded"}, status_code=400)

        new_photos = []
        rejected_files = []

        for file in photos:
            try:
                # ✅ Read and open the uploaded photo
                image_bytes = await file.read()
                size_kb = len(image_bytes) / 1024
                size_mb = size_kb / 1024
                print(f"📦 Received file: {file.filename} | Size: {size_kb:.2f} KB ({size_mb:.2f} MB)")

                image = Image.open(io.BytesIO(image_bytes))
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

                # ✅ Extract face embeddings
                embeddings = extract_faces(image)

                if not embeddings:
                    print(f"❌ No face found in: {file.filename}")
                    rejected_files.append(file.filename)
                    continue

                # ✅ Compress image
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=40, optimize=True)
                buffer.seek(0)
                compressed_image = buffer.getvalue()

                # ✅ Upload compressed image to R2
                filename = f"gallery/{uuid.uuid4().hex}.jpg"
                image_url = upload_to_r2(compressed_image, filename)

                # ✅ Prepare photo record
                photo = {
                    "photo_id": str(uuid.uuid4()),
                    "image": image_url,
                    "face_embeddings": embeddings
                }
                new_photos.append(photo)

            except Exception as e:
                print(f"❌ Failed to process {file.filename}: {e}")
                rejected_files.append(file.filename)

        # ✅ Update album in database
        if new_photos:
            update_result = albums_collection.update_one(
                {"_id": album_id},
                {
                    "$set": {"last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")},
                    "$push": {"photos": {"$each": new_photos}}
                }
            )
            print("Matched:", update_result.matched_count, "| Modified:", update_result.modified_count)
            preload_embeddings()  # ✅ reload for FastAPI
            redis_client.set("face_data_dirty", "1")  # ✅ notify queue


        return JSONResponse(content={
            "message": "Upload complete",
            "uploaded": len(new_photos),
            "rejected": rejected_files
        }, status_code=201)

    except Exception as e:
        print("❌ Upload failed:", str(e))
        return JSONResponse(content={"error": "Upload failed"}, status_code=500)



@app.post("/reload-embeddings")
async def reload_embeddings():
    try:
        total = preload_embeddings()
        return {"message": "Embeddings reloaded", "embeddings_loaded": total}
    except Exception as e:
        print("❌ Error during embedding preload:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

@search_huey.task()  # removed bind=True
def process_search_by_upload(img_bytes, submitted_time):
    from PIL import Image
    import numpy as np
    import io
    from datetime import datetime
    from embedding_store import preload_embeddings, photo_embeddings, photo_url_mapping  # optional modular import
    if redis_client.get("face_data_dirty") == b"1":
            print("🔄 Redis flag set: reloading embeddings in queue")
            preload_embeddings()
            redis_client.set("face_data_dirty", "0")

    # ✅ Expiry check
    now = datetime.utcnow()
    submitted = datetime.strptime(submitted_time, "%Y-%m-%dT%H:%M:%S")
    diff = (now - submitted).total_seconds()
    if diff > 180:
        print("❌ Task expired, skipping.")
        return {"error": "Task expired (180s limit exceeded)"}
        
    image_obj = Image.open(io.BytesIO(img_bytes))
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

    image_obj.load()  # force decoding
    file_size_kb = len(img_bytes) / 1024  # in KB
    file_size_kb = round(file_size_kb, 2)  # rounded to 2 decimals

    print(f"✅ Uploaded image file size: {file_size_kb} KB")
    print("✅ Uploaded image format:", image_obj.format, "| size:", image_obj.size, "| mode:", image_obj.mode)


    if image_obj.mode == "RGBA":
        image_obj = image_obj.convert("RGB")

    query_embeddings = extract_faces(image_obj)
    if not query_embeddings:
        return {"error": "No face found in uploaded photo"}

    matched_photo_ids = set()
    photo_ids = list(photo_embeddings.keys())
    total_photos = len(photo_ids)

    def match_batch(start, end):
        nonlocal matched_photo_ids
        for pid in photo_ids[start:end]:
            for query_face in query_embeddings:
                query_emb = np.array(query_face["embedding"])
                query_norm = np.linalg.norm(query_emb)
                for emb, emb_norm in photo_embeddings[pid]:
                    cosine_sim = np.dot(query_emb, emb) / (query_norm * emb_norm)
                    if cosine_sim > 0.75:
                        matched_photo_ids.add(pid)
                        break
                if pid in matched_photo_ids:
                    break

    # ✅ Stage 1
    match_batch(0, 16000)
    if len(matched_photo_ids) == 0:
        return {"photos": []}

    # ✅ Stage 2
    if len(matched_photo_ids) >= 16:
        match_batch(16000, 32000)
        

    # ✅ Stage 3
    if len(matched_photo_ids) >= 32:
        match_batch(32000, total_photos)
        




    if not matched_photo_ids:
            return JSONResponse(
                content={"error": "No matching faces found in database, either there is no photo of uploaded face or face is not clear"},
                status_code=404
            )

    matched_photos = [{"photo_id": pid, "image": photo_url_mapping[pid]} for pid in matched_photo_ids]
    return {"photos": matched_photos}

@app.post("/search-by-upload")
async def search_by_upload(image: UploadFile = File(...)):
    if not image or image.filename == "":
        return JSONResponse(content={"error": "No image file provided"}, status_code=400)

    img_bytes = await image.read()
    submitted_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    # ✅ Correct call with `args=...` and `delay=0`
    task = process_search_by_upload.schedule(args=(img_bytes, submitted_time), delay=0)

    return JSONResponse(content={"task_id": task.id})

@app.get("/debug/preload-status")
def preload_status():
    from datetime import datetime

    # Optional: Store last preload time globally when preload_embeddings() is called
    global last_preload_time
    return {
        "total_preloaded_photos": len(photo_embeddings),
        "sample_photo_ids": list(photo_embeddings.keys())[:5],
        "sample_image_urls": [photo_url_mapping[pid] for pid in list(photo_url_mapping.keys())[:5]],
        "last_preload_time": last_preload_time if 'last_preload_time' in globals() else "Unknown"
    }

@app.get("/task-status/{task_id}")
def get_task_status(task_id: str):
    try:
        result = search_huey.result(task_id)  # ✅ Already returns final result (dict or None)

        if result is None:
            return {"status": "pending"}  # task not yet done

        return {"status": "done", "result": result}

    except Exception as e:
        return {"status": "error", "error": str(e)}

    
@app.post("/upload-banner")
async def upload_banner(
    title: str = Form(...),
    size: str = Form(""),
    image: UploadFile = File(...)
):
    if not image or not title:
        return JSONResponse(content={"error": "Missing title or image"}, status_code=400)

    try:
        content_type = image.content_type
        ext = content_type.split("/")[-1].lower()

        if ext not in ["png", "jpg", "jpeg"]:
            return JSONResponse(content={"error": "Unsupported image format"}, status_code=400)

        filename = f"banners/{uuid.uuid4().hex}.{ext}"
        image_bytes = await image.read()

        public_url = upload_to_r2(image_bytes, filename)

        banner_id = str(uuid.uuid4())
        banners_collection.insert_one({
            "_id": banner_id,
            "title": title,
            "image": public_url,
            "size": size,
            "date": datetime.now().strftime("%d/%m/%Y"),
        })

        return {"url": public_url, "id": banner_id}

    except Exception as e:
        print("Upload error:", e)
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)



@app.delete("/delete-banner/{banner_id}")
async def delete_banner(banner_id: str):
    try:
        banners = dist_and_depart_db["banners"]
        banner = banners.find_one({"_id": banner_id})

        if not banner:
            return JSONResponse(content={"error": "Banner not found"}, status_code=404)

        key = banner["image"].split("banners/")[1]

        s3_client.delete_object(Bucket=R2_BUCKET_NAME, Key=f"banners/{key}")

        banners.delete_one({"_id": banner_id})

        return {"message": "Deleted"}

    except Exception as e:
        print("Error deleting banner:", e)
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)


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
        preload_embeddings()
        redis_client.set("face_data_dirty", "1")
        return {"message": f"Deleted {result.deleted_count} albums successfully"}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



@app.delete("/photo/{album_id}/{photo_id}")
async def delete_photo(album_id: str, photo_id: str):
    album = albums_collection.find_one({"_id": album_id}, {"photos": 1})
    if not album:
        return JSONResponse(content={"error": "Album not found"}, status_code=404)

    photo_to_delete = next((photo for photo in album.get("photos", []) if photo.get("photo_id") == photo_id), None)

    if not photo_to_delete:
        return JSONResponse(content={"error": "Photo not found"}, status_code=404)

    delete_from_r2(photo_to_delete.get("image"))

    albums_collection.update_one({"_id": album_id}, {"$pull": {"photos": {"photo_id": photo_id}}})

    preload_embeddings()
    redis_client.set("face_data_dirty", "1")

    return {"message": "Photo deleted successfully"}



