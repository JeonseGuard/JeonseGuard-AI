from crop_utils import separate_parts
from ocr_utils import get_part_1_info, get_part_2_info, get_part_3_info
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import os
import json
import io
from gcs_utils import upload_image_to_storage, upload_json_to_storage, download_json_from_storage
from google.api_core.exceptions import NotFound
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["https://jeonse-guard-frontend.vercel.app",
                                  "http://localhost:3000"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

@app.post("/upload_image")
async def upload_image(file:UploadFile = File(...)):
    try:
        image_name = os.path.splitext(file.filename)[0]

        contents = await file.read()
        bytes_io_image = io.BytesIO(contents)
        upload_image_to_storage(os.environ.get('BUCKET_NAME'), f'images/{image_name}.jpg', bytes_io_image)

        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code = 400, detail = "이미지 디코딩 실패")

        part_1_parts, part_2_parts, part_3_parts = separate_parts(image)
    
        dict = {}
        images = []
        for part in part_1_parts:
            if part is None:
                raise HTTPException(status_code = 500, detail = f"이미지 파일 로드 실패")
            images.append(part)
        dict = get_part_1_info(images, dict)

        images = []
        for part in part_2_parts:
            if part is None:
                raise HTTPException(status_code = 500, detail = f"이미지 파일 로드 실패")
            images.append(part)
        dict = get_part_2_info(images, dict)

        images = []
        for part in part_3_parts:
            if part is None:
                raise HTTPException(status_code = 500, detail = f"이미지 파일 로드 실패")
            images.append(part)
        dict = get_part_3_info(images, dict)

        json_dict = json.dumps(dict, ensure_ascii = False, indent = 4)
        bytes_io_json = io.BytesIO(json_dict.encode('utf-8'))
        upload_json_to_storage(os.environ.get('BUCKET_NAME'), f'jsons/{image_name}.json', bytes_io_json)

        return {"message": "데이터를 업로드 했습니다."}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))

@app.post("/get_part_1_json")
async def get_part_1_json(name:str = Form(...)):
    need_keys = ["address", "bun", "ji", "dongName", "floorName", "hoName",
                 "structUse", "landArea", "bldgArea", "exclArea", "deposit",
                 "downPayment", "interimPayment", "balance", "interimDate", "balanceDate"]

    try:
        dict = download_json_from_storage(os.environ.get('BUCKET_NAME'), f'jsons/{name}.json')
    except NotFound:
        raise HTTPException(status_code=404, detail="json 파일을 찾을 수 없습니다.")
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))
    
    return {key: dict[key] for key in need_keys if key in dict}

@app.post("/get_part_2_json")
async def get_part_2_json(name:str = Form(...)):
    need_keys = ["lessorName", "lessorAddress", "lesseeName", "lesseeAddress",
                 "lessorId", "lesseeId", "lessorPhone", "lesseePhone"]
    
    try:
        dict = download_json_from_storage(os.environ.get('BUCKET_NAME'), f'jsons/{name}.json')
    except NotFound:
        raise HTTPException(status_code=404, detail="json 파일을 찾을 수 없습니다.")
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))
    
    return {key: dict[key] for key in need_keys if key in dict}

@app.post("/get_part_3_json")
async def get_part_3_json(name:str = Form(...)):
    need_keys = ["clausesResponse", "scResponse"]
    
    try:
        dict = download_json_from_storage(os.environ.get('BUCKET_NAME'), f'jsons/{name}.json')
    except NotFound:
        raise HTTPException(status_code=404, detail="json 파일을 찾을 수 없습니다.")
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))
    
    return {key: dict[key] for key in need_keys if key in dict}