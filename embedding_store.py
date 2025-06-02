# embedding_store.py

import numpy as np
from datetime import datetime
from pymongo import MongoClient
import pymongo
import certifi

# MongoDB connection
client = pymongo.MongoClient(
    os.getenv("MONGO_DB_URL"),
    tls=True,
    tlsCAFile=certifi.where()
)
photo_gallery_db = client["photo_gallery"]
albums_collection = photo_gallery_db["albums"]

# Global in-memory storage
photo_embeddings = {}
photo_url_mapping = {}
last_preload_time = None

def preload_embeddings(limit=100000):
    global photo_embeddings, photo_url_mapping, last_preload_time

    print("🔄 Preloading face embeddings... (latest first, max", limit, ")")

    photo_embeddings.clear()
    photo_url_mapping.clear()

    total_loaded = 0
    albums_cursor = albums_collection.find({}).sort("last_updated", -1)

    for album in albums_cursor:
        for photo in album.get("photos", []):
            photo_id = photo.get("photo_id")
            photo_url = photo.get("image")
            embeddings = photo.get("face_embeddings", [])
            if not embeddings:
                continue

            face_list = []
            for face in embeddings:
                if total_loaded >= limit:
                    break
                emb = np.array(face.get("embedding"))
                emb_norm = np.linalg.norm(emb)
                face_list.append((emb, emb_norm))
                total_loaded += 1

            if face_list:
                photo_embeddings[photo_id] = face_list
                photo_url_mapping[photo_id] = photo_url

            if total_loaded >= limit:
                break
        if total_loaded >= limit:
            break

    last_preload_time = datetime.utcnow()
    print(f"✅ Preloaded {total_loaded} face embeddings from {len(photo_embeddings)} photos.")
    return total_loaded
