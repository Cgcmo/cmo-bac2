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
# from datetime import datetime, timedelta
from datetime import datetime, timedelta, timezone

# from deepface.DeepFace import build_model
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
import time



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
stories_collection = dist_and_depart_db["story"]
statuses_collection = dist_and_depart_db["status"]
notices_collection = dist_and_depart_db["notice"]
event_updates_collection = dist_and_depart_db["eventupdate"]
patrika_collection = dist_and_depart_db["patrika"]
videos_collection = dist_and_depart_db["videos"]
ytlive_collection = dist_and_depart_db["ytlive"]





auth_db = client["auth_db"]
users_collection = auth_db["users"]
clients_collection = auth_db["clients"]
download_count_collection = auth_db["download-count"]
visitor_collection = auth_db["visitor_logs"]
otp_collection = auth_db["otp_verifications"]

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

YT_CACHE = {
    "data": None,
    "last_fetch": 0
}


@app.get("/ytlive")
async def get_youtube_live():
    """
    Priority:
    1. Currently LIVE stream
    2. Last completed LIVE stream (automatic)
    3. NONE
    """
    now = time.time()

    if YT_CACHE["data"] and now - YT_CACHE["last_fetch"] < 60:
        return YT_CACHE["data"]

    API_KEY = os.getenv("YOUTUBE_API_KEY")
    CHANNEL_ID = os.getenv("YOUTUBE_CHANNEL_ID")

    search_url = "https://www.googleapis.com/youtube/v3/search"
    videos_url = "https://www.googleapis.com/youtube/v3/videos"

    # -------------------------------------------------
    # 1️⃣ CHECK CURRENTLY LIVE
    # -------------------------------------------------
    live_res = requests.get(search_url, params={
        "part": "snippet",
        "channelId": CHANNEL_ID,
        "eventType": "live",
        "type": "video",
        "maxResults": 1,
        "key": API_KEY
    }).json()

    if live_res.get("items"):
        v = live_res["items"][0]
        video_id = v["id"]["videoId"]

        data = {
            "status": "LIVE",
            "videoId": video_id,
            "title": v["snippet"]["title"],
            "embedUrl": f"https://www.youtube.com/embed/{video_id}"
        }

        YT_CACHE.update({"data": data, "last_fetch": now})
        return data

    # -------------------------------------------------
    # 2️⃣ FIND LAST COMPLETED LIVE (correct way)
    # -------------------------------------------------
    latest_res = requests.get(search_url, params={
        "part": "id",
        "channelId": CHANNEL_ID,
        "order": "date",
        "type": "video",
        "maxResults": 5,
        "key": API_KEY
    }).json()

    video_ids = [
        item["id"]["videoId"]
        for item in latest_res.get("items", [])
    ]

    if video_ids:
        details_res = requests.get(videos_url, params={
            "part": "snippet,liveStreamingDetails",
            "id": ",".join(video_ids),
            "key": API_KEY
        }).json()

        for v in details_res.get("items", []):
            live_details = v.get("liveStreamingDetails")
            if live_details and live_details.get("actualEndTime"):
                video_id = v["id"]

                data = {
                    "status": "RECORDED",
                    "videoId": video_id,
                    "title": v["snippet"]["title"],
                    "embedUrl": f"https://www.youtube.com/embed/{video_id}"
                }

                YT_CACHE.update({"data": data, "last_fetch": now})
                return data

    # -------------------------------------------------
    # 3️⃣ NOTHING FOUND
    # -------------------------------------------------
    data = {"status": "NONE"}
    YT_CACHE.update({"data": data, "last_fetch": now})
    return data

# @app.get("/ytlive")
# async def get_youtube_live():
#     """
#     YouTube Live Status API
#     Returns:
#     - LIVE
#     - ENDED
#     - NONE
#     """
#     now = time.time()

#     # Cache for 60 seconds
#     if YT_CACHE["data"] and now - YT_CACHE["last_fetch"] < 60:
#         return YT_CACHE["data"]

#     API_KEY = os.getenv("YOUTUBE_API_KEY")
#     CHANNEL_ID = os.getenv("YOUTUBE_CHANNEL_ID")

#     search_url = "https://www.googleapis.com/youtube/v3/search"

#     # 1️⃣ CHECK LIVE
#     live_res = requests.get(search_url, params={
#         "part": "snippet",
#         "channelId": CHANNEL_ID,
#         "eventType": "live",
#         "type": "video",
#         "maxResults": 1,
#         "key": API_KEY
#     }).json()

#     if live_res.get("items"):
#         v = live_res["items"][0]
#         video_id = v["id"]["videoId"]

#         data = {
#             "status": "LIVE",
#             "videoId": video_id,
#             "title": v["snippet"]["title"],
#             "embedUrl": f"https://www.youtube.com/embed/{video_id}"
#         }

#         YT_CACHE.update({"data": data, "last_fetch": now})
#         return data

#     # 2️⃣ CHECK IF RECENT LIVE JUST ENDED (last 5 minutes)
#     recent_res = requests.get(search_url, params={
#         "part": "snippet",
#         "channelId": CHANNEL_ID,
#         "order": "date",
#         "type": "video",
#         "maxResults": 1,
#         "key": API_KEY
#     }).json()

#     if recent_res.get("items"):
#         published = recent_res["items"][0]["snippet"]["publishedAt"]
#         published_time = datetime.fromisoformat(published.replace("Z", "+00:00"))

#         if datetime.now(timezone.utc) - published_time < timedelta(minutes=5):
#             data = {"status": "ENDED"}
#             YT_CACHE.update({"data": data, "last_fetch": now})
#             return data

#     # 3️⃣ NO LIVE
#     data = {"status": "NONE"}
#     YT_CACHE.update({"data": data, "last_fetch": now})
#     return data


    
# YT_CACHE = {
#     "data": None,
#     "last_fetch": 0
# }

# @app.get("/ytlive")
# async def get_youtube_live():
#     """
#     Priority:
#     1. Live running
#     2. Upcoming (scheduled)
#     3. Latest uploaded video

#     Cache: 60 seconds
#     """
#     now = time.time()

#     # ✅ Cache for 60 seconds
#     if YT_CACHE["data"] and now - YT_CACHE["last_fetch"] < 60:
#         return YT_CACHE["data"]

#     API_KEY = os.getenv("YOUTUBE_API_KEY")
#     CHANNEL_ID = os.getenv("YOUTUBE_CHANNEL_ID")

#     search_url = "https://www.googleapis.com/youtube/v3/search"

#     # =========================================================
#     # 🔴 1. CHECK CURRENTLY LIVE
#     # =========================================================
#     live_params = {
#         "part": "snippet",
#         "channelId": CHANNEL_ID,
#         "eventType": "live",
#         "type": "video",
#         "maxResults": 1,
#         "key": API_KEY
#     }

#     live_res = requests.get(search_url, params=live_params).json()

#     if live_res.get("items"):
#         v = live_res["items"][0]
#         video_id = v["id"]["videoId"]

#         data = [{
#             "status": "live",
#             "title": v["snippet"]["title"],
#             "videoId": video_id,
#             "watchUrl": f"https://www.youtube.com/watch?v={video_id}",
#             "embedUrl": f"https://www.youtube.com/embed/{video_id}",
#             "image": v["snippet"]["thumbnails"]["high"]["url"]
#         }]

#         YT_CACHE.update({"data": data, "last_fetch": now})
#         return data

    # =========================================================
    # 🟡 2. CHECK UPCOMING (SCHEDULED LIVE)
    # =========================================================
    upcoming_params = {
        "part": "snippet",
        "channelId": CHANNEL_ID,
        "eventType": "upcoming",
        "type": "video",
        "maxResults": 1,
        "key": API_KEY
    }

    upcoming_res = requests.get(search_url, params=upcoming_params).json()

    if upcoming_res.get("items"):
        v = upcoming_res["items"][0]
        video_id = v["id"]["videoId"]

        data = [{
            "status": "upcoming",
            "title": v["snippet"]["title"],
            "videoId": video_id,
            "watchUrl": f"https://www.youtube.com/watch?v={video_id}",
            "embedUrl": f"https://www.youtube.com/embed/{video_id}",
            "image": v["snippet"]["thumbnails"]["high"]["url"]
        }]

        YT_CACHE.update({"data": data, "last_fetch": now})
        return data

    # =========================================================
    # ⚪ 3. FALLBACK → LATEST UPLOADED VIDEO
    # =========================================================
    latest_params = {
        "part": "snippet",
        "channelId": CHANNEL_ID,
        "order": "date",
        "type": "video",
        "maxResults": 1,
        "key": API_KEY
    }

    latest_res = requests.get(search_url, params=latest_params).json()

    if latest_res.get("items"):
        v = latest_res["items"][0]
        video_id = v["id"]["videoId"]

        data = [{
            "status": "video",
            "title": v["snippet"]["title"],
            "videoId": video_id,
            "watchUrl": f"https://www.youtube.com/watch?v={video_id}",
            "embedUrl": f"https://www.youtube.com/embed/{video_id}",
            "image": v["snippet"]["thumbnails"]["high"]["url"]
        }]

        YT_CACHE.update({"data": data, "last_fetch": now})
        return data

    return []

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


# ========== Get All Albums ==========
# ========== Get All Albums ==========
@app.get("/albums")
async def get_albums(
    page: int = 1,
    limit: int = 16,
    districts: str = "",
    departments: str = "",
    from_date: str = "",
    to_date: str = "",
    with_cm: str = "" 
):
    try:
        skip = (page - 1) * limit

        # Build query
        query = {}

        if districts:
            query["districts"] = {"$in": districts.split(",")}

        if departments:
            query["department"] = {"$in": departments.split(",")}
        
        if with_cm:   # 👈 Filter by CM category
            query["with_cm"] = with_cm

        # ✅ Use YYYY-MM-DD directly (same format saved in DB when creating albums)
        if from_date and to_date:
            query["date"] = {"$gte": from_date, "$lte": to_date}

        total_count = albums_collection.count_documents(query)

        albums = list(albums_collection.aggregate([
            {"$match": query},
            {"$skip": skip},
            {"$limit": limit},
            {"$project": {
                "_id": 1,
                "name": 1,
                "date": 1,
                "cover": 1,
                "districts": 1,
                "department": 1,
                "photo_count": {"$size": {"$ifNull": ["$photos", []]}}
            }}
        ]))

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


@app.get("/get-banners")
async def get_banners():
    banners = list(dist_and_depart_db["banners"].find({}, {"_id": 1, "title": 1, "image": 1, "size": 1, "date": 1}))
    formatted = [{"id": str(b["_id"]), **b} for b in banners]
    return formatted



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

@app.post("/delete-albums")
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

    return {"message": "Photo deleted successfully"}


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

# @app.post("/master-search")
# async def master_search(data: MasterSearchRequest):
#     query = data.query.strip().lower()

#     if not query:
#         return JSONResponse(content={"error": "Empty search"}, status_code=400)

#     matching_photos = []

#     albums = albums_collection.find()

#     for album in albums:
#         # Simple case-insensitive matching
#         album_name = album.get("name", "").lower()
#         department = album.get("department", "").lower()
#         districts = [d.lower() for d in album.get("districts", [])]

#         if (
#             query in album_name or
#             query in department or
#             any(query in d for d in districts)
#         ):
#             matched_by = []
#             if query in album_name:
#                 matched_by.append("Event")
#             if query in department:
#                 matched_by.append("Department")
#             if any(query in d for d in districts):
#                 matched_by.append("District")

#             for photo in album.get("photos", []):
#                 matching_photos.append({
#                     "photo_id": photo["photo_id"],
#                     "image": photo["image"],
#                     "matched_by": matched_by,
#                     "album_name": album.get("name", ""),
#                     "department": album.get("department", ""),
#                     "districts": album.get("districts", [])
#                 })

#     return {"photos": matching_photos}


@app.post("/master-search")
async def master_search(data: MasterSearchRequest):
    query = data.query.strip().lower()
    if not query:
        return JSONResponse(content={"error": "Empty search"}, status_code=400)

    matching_albums = []

    albums = albums_collection.find()

    for album in albums:
        album_name = album.get("name", "").lower()
        department = album.get("department", "").lower()
        districts = [d.lower() for d in album.get("districts", [])]

        if (
            query in album_name or
            query in department or
            any(query in d for d in districts)
        ):
            matching_albums.append({
                "album_id": str(album["_id"]),
                "name": album.get("name", ""),
                "cover": album.get("cover", ""),
                "date": album.get("date", ""),
                "department": album.get("department", ""),
                "districts": album.get("districts", []),
                "photo_count": len(album.get("photos", []))
            })

    return {"albums": matching_albums}

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
async def update_user(
    user_id: str,
    name: str = Form(None),
    mobile: str = Form(None),
    district: str = Form(None),
    photo: UploadFile = File(None),
):
    update_fields = {}

    if name:
        update_fields["name"] = name
    if mobile:
        update_fields["mobile"] = mobile
    if district:
        update_fields["district"] = district

    # 🔥 PHOTO UPDATE
    if photo:
        file_bytes = await photo.read()
        ext = photo.filename.split(".")[-1].lower()
        filename = f"profile/{uuid.uuid4().hex}.{ext}"

        image_url = upload_to_r2(file_bytes, filename)
        update_fields["photo"] = image_url

    if not update_fields:
        return JSONResponse(
            content={"error": "No fields to update"},
            status_code=400
        )

    result = users_collection.update_one(
        {"_id": user_id},
        {"$set": update_fields}
    )

    if result.modified_count == 0:
        return JSONResponse(
            content={"error": "User not found"},
            status_code=404
        )

    return {"message": "Profile updated successfully"}
# @app.put("/update-user/{user_id}")
# async def update_user(user_id: str, data: UpdateUserRequest):
#     update_fields = {}

#     for field in ["name", "email", "mobile", "district", "status"]:
#         value = getattr(data, field)
#         if value is not None:
#             update_fields[field] = value

#     if not update_fields:
#         return JSONResponse(content={"error": "No fields to update"}, status_code=400)

#     result = users_collection.update_one({"_id": user_id}, {"$set": update_fields})

#     if result.modified_count == 0:
#         result = clients_collection.update_one({"_id": user_id}, {"$set": update_fields})
#         if result.modified_count == 0:
#             return JSONResponse(content={"error": "User not found or no changes made"}, status_code=404)

#     return {"message": "User updated successfully"}


# @app.get("/users")
# async def get_users():
#     projection = {"name": 1, "email": 1, "mobile": 1, "district": 1, "role": 1, "status": 1}

#     users = list(users_collection.find({}, projection))
#     clients = list(clients_collection.find({}, projection))

#     combined_users = users + clients

#     for user in combined_users:
#         user["_id"] = str(user["_id"])
#         user["role"] = user.get("role", "User")
#         user["status"] = user.get("status", True)
#         user["mobile"] = user.get("mobile") or "Gmail User"
#         user["district"] = user.get("district") or "Gmail User"

#     return combined_users


# @app.get("/users")
# async def get_users(
#     filter: str = Query("All"),
#     page: int = Query(1, ge=1),
#     limit: int = Query(10, ge=1, le=100),
#      search: str = Query(None),  
#     mobile: str = Query(None)    
# ):
#     projection = {"name": 1, "email": 1, "mobile": 1, "district": 1, "role": 1, "status": 1}

#     users = list(users_collection.find({}, projection))
#     clients = list(clients_collection.find({}, projection))

#     combined_users = users + clients

#     # Normalize fields
#     for user in combined_users:
#         user["_id"] = str(user["_id"])
#         user["role"] = user.get("role", "User")
#         user["status"] = user.get("status", True)
#         user["mobile"] = user.get("mobile") or "Gmail User"
#         user["district"] = user.get("district") or "Gmail User"

#     # ✅ Apply filter server-side
#     if filter == "Admin":
#         combined_users = [u for u in combined_users if u["role"] == "Admin"]
#     elif filter == "User":
#         combined_users = [u for u in combined_users if u["role"] == "User"]
#     elif filter == "Limited Access":
#         combined_users = [u for u in combined_users if not u["status"]]

#      if search:
#         combined_users = [u for u in combined_users if search.lower() in u["name"].lower()]
#     # if mobile:
#     #     combined_users = [u for u in combined_users if str(u["mobile"]) == str(mobile)]
#     if mobile:
#     combined_users = [u for u in combined_users if mobile in str(u["mobile"])]



#     total = len(combined_users)

#     # ✅ Apply pagination
#     start = (page - 1) * limit
#     end = start + limit
#     paginated = combined_users[start:end]

#     return {"users": paginated, "total": total}


@app.get("/users")
async def get_users(
    filter: str = Query("All"),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    search: str = Query(None),
    mobile: str = Query(None)
):
    projection = {"name": 1, "email": 1, "mobile": 1, "district": 1, "role": 1, "status": 1,"photo": 1,}

    users = list(users_collection.find({}, projection))
    clients = list(clients_collection.find({}, projection))

    combined_users = users + clients

    # Normalize fields
    for user in combined_users:
        user["_id"] = str(user["_id"])
        user["role"] = user.get("role", "User")
        user["status"] = user.get("status", True)
        user["mobile"] = user.get("mobile") or "Gmail User"
        user["district"] = user.get("district") or "Gmail User"

    # ✅ Apply filter server-side
    if filter == "Admin":
        combined_users = [u for u in combined_users if u["role"] == "Admin"]
    elif filter == "User":
        combined_users = [u for u in combined_users if u["role"] == "User"]
    elif filter == "Limited Access":
        combined_users = [u for u in combined_users if not u["status"]]

    # if search:
    #     combined_users = [u for u in combined_users if search.lower() in u["name"].lower()]

    # if mobile:
    #     combined_users = [u for u in combined_users if mobile in str(u["mobile"])]

    if search:
        combined_users = [
        u for u in combined_users
        if search.lower() in u["name"].lower() or search in str(u["mobile"])
    ]


    total = len(combined_users)

    # ✅ Apply pagination
    start = (page - 1) * limit
    end = start + limit
    paginated = combined_users[start:end]

    return {"users": paginated, "total": total}


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


@app.put("/update-client/{user_id}")
async def update_client(
    user_id: str,
    name: str = Form(None),
    mobile: str = Form(None),
    district: str = Form(None),
    photo: UploadFile = File(None)
):
    try:
        update_fields = {}

        if name:
            update_fields["name"] = name

        if mobile:
            update_fields["mobile"] = mobile

        if district:
            update_fields["district"] = district

        # Upload profile photo
        if photo:
            file_bytes = await photo.read()

            ext = photo.filename.split(".")[-1].lower()

            filename = f"profile/{uuid.uuid4().hex}.{ext}"

            image_url = upload_to_r2(
                file_bytes,
                filename
            )

            update_fields["photo"] = image_url

        if not update_fields:
            return JSONResponse(
                content={"error": "No fields to update"},
                status_code=400
            )

        result = clients_collection.update_one(
            {"_id": user_id},
            {"$set": update_fields}
        )

        if result.matched_count == 0:
            return JSONResponse(
                content={"error": "Client not found"},
                status_code=404
            )

        return {
            "message": "Profile updated successfully"
        }

    except Exception as e:
        print("❌ Update Client Error:", e)

        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
        
# class UpdateClientRequest(BaseModel):
#     name: str = None
#     mobile: str = None
#     district: str = None

# @app.put("/update-client/{user_id}")
# async def update_client(user_id: str, data: UpdateClientRequest):
#     allowed_fields = ["name", "mobile", "district"]

#     update_fields = {}

#     for field in allowed_fields:
#         value = getattr(data, field)
#         if value is not None:
#             update_fields[field] = value

#     if not update_fields:
#         return JSONResponse(content={"error": "No valid fields to update"}, status_code=400)

#     result = clients_collection.update_one({"_id": user_id}, {"$set": update_fields})

#     if result.modified_count == 0:
#         return JSONResponse(content={"error": "Client not found or no changes made"}, status_code=404)

#     return {"message": "Client updated successfully"}


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




@app.get("/filters")
async def get_filters():
    try:
        # fetch distinct districts from albums OR from the dedicated collection
        districts = [d["name"] for d in districts_collection.find({}, {"name": 1})]

        # fetch distinct departments
        departments = [dep["name"] for dep in departments_collection.find({}, {"name": 1})]

        # remove duplicates and sort
        districts = sorted(list(set(districts)))
        departments = sorted(list(set(departments)))

        return {
            "districts": districts,
            "departments": departments
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


class StoryUpdateRequest(BaseModel):
    title: str = None
    date: str = None
    desc: str = None
    image: str = None


from fastapi import Depends

async def admin_required(request: Request):
    user_id = request.headers.get("X-User-Id")
    if not user_id:
        return JSONResponse(content={"error": "Unauthorized"}, status_code=401)

    user = users_collection.find_one({"_id": user_id})
    if not user or user.get("role") != "Admin":
        return JSONResponse(content={"error": "Forbidden"}, status_code=403)
    return user




# ---------- Create Status (Admin only) ----------
# @app.post("/status")
# async def create_status(
#     title: str = Form(...),
#     image: UploadFile = File(...),
#     user=Depends(admin_required)
# ):
#     try:
#         file_bytes = await image.read()
#         filename = f"status/{uuid.uuid4().hex}.jpg"
#         image_url = upload_to_r2(file_bytes, filename)

#         status_id = str(uuid.uuid4())
#         status_doc = {
#             "_id": status_id,
#             "title": title,
#             "image": image_url
#         }
#         statuses_collection.insert_one(status_doc)

#         return {"id": status_id, "url": image_url, "message": "Status created successfully"}
#     except Exception as e:
#         print("❌ Error creating status:", str(e))
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# ---------- Create Multiple Status (Admin only) ----------
@app.post("/status")
async def create_multiple_status(
    title: str = Form(...),
    images: list[UploadFile] = File(...),
    user=Depends(admin_required)
):
    try:
        created_statuses = []

        for image in images:
            file_bytes = await image.read()

            ext = image.filename.split(".")[-1].lower()
            if ext not in ["jpg", "jpeg", "png"]:
                continue

            filename = f"status/{uuid.uuid4().hex}.{ext}"
            image_url = upload_to_r2(file_bytes, filename)

            status_id = str(uuid.uuid4())
            status_doc = {
                "_id": status_id,
                "title": title,
                "image": image_url
            }

            statuses_collection.insert_one(status_doc)

            created_statuses.append({
                "id": status_id,
                "url": image_url
            })

        return {
            "message": f"{len(created_statuses)} statuses created",
            "statuses": created_statuses
        }

    except Exception as e:
        print("❌ Error creating statuses:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)



# ---------- Get Status (Public) ----------
@app.get("/status")
async def get_status():
    statuses = list(statuses_collection.find({}, {"_id": 1, "title": 1, "image": 1}))
    formatted = [
        {
            "id": str(s["_id"]),
            "title": s.get("title", ""),
            "image": s.get("image", "")
        }
        for s in statuses
    ]
    return formatted



# ---------- Delete Status (Admin only) ----------
@app.delete("/status/{status_id}")
async def delete_status(status_id: str, user=Depends(admin_required)):
    status = statuses_collection.find_one({"_id": status_id})
    if not status:
        return JSONResponse(content={"error": "Status not found"}, status_code=404)

    delete_from_r2(status.get("image"))
    statuses_collection.delete_one({"_id": status_id})
    return {"message": "Status deleted successfully"}



# ---------- Create Notice (Admin only) ----------
@app.post("/notices")
async def create_notice(
    title: str = Form(...),
    date: str = Form(...),
    pdf: UploadFile = File(...),
    user=Depends(admin_required)
):
    try:
        if pdf.content_type != "application/pdf":
            return JSONResponse(content={"error": "Only PDF files allowed"}, status_code=400)

        file_bytes = await pdf.read()
        filename = f"pdf/{uuid.uuid4().hex}.pdf"
        pdf_url = upload_to_r2(file_bytes, filename)

        notice_id = str(uuid.uuid4())
        notice_doc = {
            "_id": notice_id,
            "title": title,
            "date": date,
            "pdf": pdf_url
        }
        notices_collection.insert_one(notice_doc)

        return {"id": notice_id, "url": pdf_url, "message": "Notice created successfully"}
    except Exception as e:
        print("❌ Error creating notice:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ---------- Get Notices (Public) ----------
@app.get("/notices")
async def get_notices():
    notices = list(notices_collection.find({}, {"_id": 1, "title": 1, "date": 1, "pdf": 1}))
    formatted = [{"id": str(n["_id"]), **n} for n in notices]
    return formatted


# ---------- Delete Notice (Admin only) ----------
@app.delete("/notices/{notice_id}")
async def delete_notice(notice_id: str, user=Depends(admin_required)):
    notice = notices_collection.find_one({"_id": notice_id})
    if not notice:
        return JSONResponse(content={"error": "Notice not found"}, status_code=404)

    delete_from_r2(notice.get("pdf"))
    notices_collection.delete_one({"_id": notice_id})
    return {"message": "Notice deleted successfully"}


# ========== Event Updates ==========
@app.post("/event-updates")
async def create_event_update(
    title_hi: str = Form(""),
    title_en: str = Form(""),
    date: str = Form(...),
    desc_hi: str = Form(""),
    desc_en: str = Form(""),
    image: UploadFile = File(...),
    user=Depends(admin_required)
):
    try:
        if not title_hi and not title_en:
            return JSONResponse(
                content={"error": "At least one title is required"},
                status_code=400
            )

        if not desc_hi and not desc_en:
            return JSONResponse(
                content={"error": "At least one description is required"},
                status_code=400
            )

        file_bytes = await image.read()
        filename = f"event_updates/{uuid.uuid4().hex}.jpg"
        image_url = upload_to_r2(file_bytes, filename)

        event_id = str(uuid.uuid4())
        doc = {
            "_id": event_id,
            "title_hi": title_hi,
            "title_en": title_en,
            "desc_hi": desc_hi,
            "desc_en": desc_en,
            "date": date,
            "image": image_url,
        }

        event_updates_collection.insert_one(doc)

        return {
            "id": event_id,
            "url": image_url,
            "message": "Event update created successfully"
        }

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
@app.put("/event-updates/{event_id}")
async def update_event_update(
    event_id: str,
    title_hi: str = Form(""),
    title_en: str = Form(""),
    desc_hi: str = Form(""),
    desc_en: str = Form(""),
    date: str = Form(""),
    image: UploadFile = File(None),
    user=Depends(admin_required)
):
    try:
        existing = event_updates_collection.find_one({"_id": event_id})
        if not existing:
            return JSONResponse(
                content={"error": "Event not found"},
                status_code=404
            )

        update_data = {
            "title_hi": title_hi,
            "title_en": title_en,
            "desc_hi": desc_hi,
            "desc_en": desc_en,
            "date": date,
        }

        # If new image uploaded
        if image:
            file_bytes = await image.read()
            filename = f"event_updates/{uuid.uuid4().hex}.jpg"
            image_url = upload_to_r2(file_bytes, filename)
            update_data["image"] = image_url

        event_updates_collection.update_one(
            {"_id": event_id},
            {"$set": update_data}
        )

        return {"message": "Event updated successfully"}

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

# @app.post("/event-updates")
# async def create_event_update(
#    title_hi: str = Form(""),
#     title_en: str = Form(""),
#     date: str = Form(...),
#     desc_hi: str = Form(""),
#     desc_en: str = Form(""),
#     image: UploadFile = File(...),
#     user=Depends(admin_required)
# ):
#     try:
#                 if not title_hi and not title_en:
#             return JSONResponse(
#                 content={"error": "At least one title is required"},
#                 status_code=400
#             )

#         if not desc_hi and not desc_en:
#             return JSONResponse(
#                 content={"error": "At least one description is required"},
#                 status_code=400
#             )
#         file_bytes = await image.read()
#         filename = f"event_updates/{uuid.uuid4().hex}.jpg"
#         image_url = upload_to_r2(file_bytes, filename)

#         event_id = str(uuid.uuid4())
#         doc = {
#             "_id": event_id,
#             "title_hi": title_hi,
#             "title_en": title_en,
#             "desc_hi": desc_hi,
#             "desc_en": desc_en,
#             "date": date,
#             "image": image_url,
#         }
#         event_updates_collection.insert_one(doc)

#         return {"id": event_id, "url": image_url, "message": "Event update created successfully"}
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)



@app.get("/event-updates")
async def get_event_updates():
    updates = list(event_updates_collection.find({}, {"_id": 1, "title_hi":1,  "title_en":1,"desc_hi":1,"desc_en":1, "date": 1, "image": 1}))
    return [{"id": str(u["_id"]), **u} for u in updates]


@app.delete("/event-updates/{event_id}")
async def delete_event_update(event_id: str, user=Depends(admin_required)):
    event = event_updates_collection.find_one({"_id": event_id})
    if not event:
        return JSONResponse(content={"error": "Event update not found"}, status_code=404)

    delete_from_r2(event.get("image"))
    event_updates_collection.delete_one({"_id": event_id})
    return {"message": "Event update deleted successfully"}


# ---------- Create Patrika (Admin only) ----------
@app.post("/patrika")
async def create_patrika(
    title: str = Form(...),
    date: str = Form(...),
    image: UploadFile = File(...),
    pdf: UploadFile = File(...),
    user=Depends(admin_required)
):
    try:
        # Upload image
        img_bytes = await image.read()
        img_filename = f"patrika/{uuid.uuid4().hex}.jpg"
        img_url = upload_to_r2(img_bytes, img_filename)

        # Upload PDF
        if pdf.content_type != "application/pdf":
            return JSONResponse(content={"error": "Only PDF allowed"}, status_code=400)
        pdf_bytes = await pdf.read()
        pdf_filename = f"patrika/{uuid.uuid4().hex}.pdf"
        pdf_url = upload_to_r2(pdf_bytes, pdf_filename)

        patrika_id = str(uuid.uuid4())
        patrika_doc = {
            "_id": patrika_id,
            "title": title,
            "date": date,
            "image": img_url,
            "pdf": pdf_url
        }
        patrika_collection.insert_one(patrika_doc)

        return {"id": patrika_id, "imageUrl": img_url, "pdfUrl": pdf_url}
    except Exception as e:
        print("❌ Error creating patrika:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ---------- Get Patrika (Public) ----------
@app.get("/patrika")
async def get_patrika():
    pat = list(patrika_collection.find({}, {"_id": 1, "title": 1, "date": 1, "image": 1, "pdf": 1}))
    return [{"id": str(p["_id"]), **p} for p in pat]


# ---------- Delete Patrika (Admin only) ----------
@app.delete("/patrika/{patrika_id}")
async def delete_patrika(patrika_id: str, user=Depends(admin_required)):
    doc = patrika_collection.find_one({"_id": patrika_id})
    if not doc:
        return JSONResponse(content={"error": "Not found"}, status_code=404)

    delete_from_r2(doc.get("image"))
    delete_from_r2(doc.get("pdf"))
    patrika_collection.delete_one({"_id": patrika_id})
    return {"message": "Patrika deleted successfully"}

@app.put("/patrika/{patrika_id}")
async def update_patrika(
    patrika_id: str,
    title: str = Form(...),
    date: str = Form(...),
    image: UploadFile = File(None),
    pdf: UploadFile = File(None),
    user=Depends(admin_required)
):
    try:
        patrika = patrika_collection.find_one({"_id": patrika_id})

        if not patrika:
            return JSONResponse(
                content={"error": "Patrika not found"},
                status_code=404
            )

        update_data = {
            "title": title,
            "date": date
        }

        # Replace image
        if image:
            delete_from_r2(patrika.get("image"))

            img_bytes = await image.read()

            img_filename = f"patrika/{uuid.uuid4().hex}.jpg"

            img_url = upload_to_r2(
                img_bytes,
                img_filename
            )

            update_data["image"] = img_url

        # Replace PDF
        if pdf:

            if pdf.content_type != "application/pdf":
                return JSONResponse(
                    content={"error": "Only PDF allowed"},
                    status_code=400
                )

            delete_from_r2(patrika.get("pdf"))

            pdf_bytes = await pdf.read()

            pdf_filename = f"patrika/{uuid.uuid4().hex}.pdf"

            pdf_url = upload_to_r2(
                pdf_bytes,
                pdf_filename
            )

            update_data["pdf"] = pdf_url

        patrika_collection.update_one(
            {"_id": patrika_id},
            {"$set": update_data}
        )

        return {
            "message": "Patrika updated successfully"
        }

    except Exception as e:
        print("❌ Patrika update error:", str(e))

        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

    
# ---------- Create Video (Admin only) ----------
@app.post("/videos")
async def create_video(
    title: str = Form(...),
    desc: str = Form(...),
    link: str = Form(...),              # embedded link (e.g. YouTube embed url)
    image: UploadFile = File(...),      # cover image
    user=Depends(admin_required)
):
    try:
        # Upload cover image to R2/videos/
        file_bytes = await image.read()
        filename = f"videos/{uuid.uuid4().hex}.jpg"
        image_url = upload_to_r2(file_bytes, filename)

        video_id = str(uuid.uuid4())
        video_doc = {
            "_id": video_id,
            "title": title,
            "desc": desc,
            "link": link,
            "image": image_url,
        }
        videos_collection.insert_one(video_doc)

        return {"id": video_id, "imageUrl": image_url, "message": "Video created successfully"}

    except Exception as e:
        print("❌ Error creating video:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ---------- Get Videos (Public) ----------
@app.get("/videos")
async def get_videos():
    videos = list(videos_collection.find({}, {"_id": 1, "title": 1, "desc": 1, "link": 1, "image": 1}))
    return [{"id": str(v["_id"]), **v} for v in videos]


# ---------- Delete Video (Admin only) ----------
@app.delete("/videos/{video_id}")
async def delete_video(video_id: str, user=Depends(admin_required)):
    doc = videos_collection.find_one({"_id": video_id})
    if not doc:
        return JSONResponse(content={"error": "Video not found"}, status_code=404)

    # delete cover image from R2
    delete_from_r2(doc.get("image"))
    videos_collection.delete_one({"_id": video_id})
    return {"message": "Video deleted successfully"}

@app.put("/videos/{video_id}")
async def update_video(
    video_id: str,
    title: str = Form(...),
    desc: str = Form(...),
    link: str = Form(...),
    image: UploadFile = File(None),
    user=Depends(admin_required)
):
    try:
        video = videos_collection.find_one({"_id": video_id})

        if not video:
            return JSONResponse(
                content={"error": "Video not found"},
                status_code=404
            )

        update_data = {
            "title": title,
            "desc": desc,
            "link": link
        }

        # Replace image if uploaded
        if image:
            delete_from_r2(video.get("image"))

            img_bytes = await image.read()

            filename = f"videos/{uuid.uuid4().hex}.jpg"

            image_url = upload_to_r2(
                img_bytes,
                filename
            )

            update_data["image"] = image_url

        videos_collection.update_one(
            {"_id": video_id},
            {"$set": update_data}
        )

        return {
            "message": "Video updated successfully"
        }

    except Exception as e:
        print("❌ Video update error:", str(e))

        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


# ---------- Create Live Stream (Admin only) ----------
# @app.post("/ytlive")
# async def create_live_stream(
#     title: str = Form(...),
#     link: str = Form(...),
#     image: UploadFile = File(...),
#     status: bool = Form(True),   # default True
#     user=Depends(admin_required)
# ):
#     try:
#         # ✅ Upload image to R2
#         file_bytes = await image.read()
#         filename = f"ytlive/{uuid.uuid4().hex}.jpg"
#         image_url = upload_to_r2(file_bytes, filename)

#         # ✅ If status is True, set all others to False
#         if status:
#             ytlive_collection.update_many({}, {"$set": {"status": False}})

#         live_id = str(uuid.uuid4())
#         live_doc = {
#             "_id": live_id,
#             "title": title,
#             "link": link,
#             "image": image_url,
#             "status": status,
#             "createdAt": datetime.utcnow()
#         }
#         ytlive_collection.insert_one(live_doc)

#         return {"id": live_id, "url": image_url, "message": "Live stream added successfully"}
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)



# # ---------- Get Live Streams (Public) ----------
# @app.get("/ytlive")
# async def get_live_streams():
#     streams = list(ytlive_collection.find({}, {"_id": 1, "title": 1, "link": 1, "image": 1, "status": 1, "createdAt": 1}))
#     return [{"id": str(s["_id"]), **s} for s in streams]


# ---------- Delete Live Stream (Admin only) ----------
@app.delete("/ytlive/{stream_id}")
async def delete_live_stream(stream_id: str, user=Depends(admin_required)):
    doc = ytlive_collection.find_one({"_id": stream_id})
    if not doc:
        return JSONResponse(content={"error": "Stream not found"}, status_code=404)

    ytlive_collection.delete_one({"_id": stream_id})
    return {"message": "Live stream deleted successfully"}


@app.put("/ytlive/{stream_id}/status")
async def update_live_status(stream_id: str, status: bool = Form(...), user=Depends(admin_required)):
    try:
        if status:
            # ✅ deactivate all other streams
            ytlive_collection.update_many({}, {"$set": {"status": False}})

        result = ytlive_collection.update_one(
            {"_id": stream_id},
            {"$set": {"status": status}}
        )

        if result.matched_count == 0:
            return JSONResponse(content={"error": "Stream not found"}, status_code=404)

        return {"message": "Status updated successfully"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)




def send_sms_otp(mobile: str, otp: str):
    url = os.getenv("SMS_API_URL")

    params = {
        "username": os.getenv("SMS_USERNAME"),
        "password": os.getenv("SMS_PASSWORD"),
        "sender": os.getenv("SMS_SENDER"),
        "to": mobile,   # 👈 DO NOT prefix +91 here
        "message": f"{otp} is the OTP to verify your mobile number with HamarCm. It will expire by today",
        "reqid": str(int(time.time())),
        "format": "json",
        "route_id": os.getenv("SMS_ROUTE_ID"),
        "Template_ID": os.getenv("SMS_TEMPLATE_ID"),
        "PE_ID": os.getenv("SMS_PE_ID"),
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        return response.text   # 👈 IMPORTANT: text, not json()
    except Exception as e:
        return str(e)



class SendOtpRequest(BaseModel):
    mobile: str


@app.post("/send-otp")
async def send_otp(data: SendOtpRequest):

    otp = str(uuid.uuid4().int)[0:6]

    # Delete old OTP if exists
    otp_collection.delete_many({"mobile": data.mobile})

    # Save new OTP
    otp_collection.insert_one({
        "mobile": data.mobile,
        "otp": otp,
        "created_at": datetime.utcnow()
    })

    sms_response = send_sms_otp(data.mobile, otp)

    return {
        "message": "OTP sent",
        "sms_response": sms_response
    }

class VerifyOtpRequest(BaseModel):
    mobile: str
    otp: str


@app.post("/verify-otp")
async def verify_otp(data: VerifyOtpRequest):

    record = otp_collection.find_one({
        "mobile": data.mobile,
        "otp": data.otp
    })

    if not record:
        return JSONResponse(
            content={"error": "Invalid OTP"},
            status_code=400
        )

    # If found → delete OTP
    otp_collection.delete_one({"_id": record["_id"]})

    return {"message": "OTP verified successfully"}
