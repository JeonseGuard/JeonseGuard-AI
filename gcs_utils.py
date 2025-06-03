from google.cloud import storage
import numpy as np
import cv2
import json

def upload_image_to_storage(bucket_name, file_path, bytes_io_image):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    
    blob.upload_from_file(bytes_io_image, content_type = 'image/jpeg')
    return "업로드 완료"

def download_image_from_storage(bucket_name, file_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    
    bytes_image = blob.download_as_bytes()
    nparr = np.frombuffer(bytes_image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def upload_json_to_storage(bucket_name, file_path, bytes_io_json):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    blob.upload_from_file(bytes_io_json, content_type = 'application/json')
    return "업로드 완료"

def download_json_from_storage(bucket_name, file_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    bytes_json = blob.download_as_bytes()
    json_dict = bytes_json.decode('utf-8')
    dict = json.loads(json_dict)
    return dict