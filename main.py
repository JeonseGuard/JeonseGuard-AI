from crop_image import separate_parts
from ocr_utils import get_part_1_info, get_part_2_info, get_part_3_info
from fastapi import FastAPI, File, UploadFile, Form
import numpy as np
import cv2
import os
import json

app = FastAPI()

@app.post("/upload_image")
async def upload_image(file:UploadFile = File(...)):
    image_name = os.path.splitext(file.filename)[0]
    images_path = './data/images'

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite(f'{images_path}/{image_name}.jpg', image)

    separate_parts(f'{images_path}/{image_name}.jpg')
    seperate_path = f'{images_path}/{image_name}'
    
    dict = {}
    part_paths = [f'{seperate_path}/part_1_{i}.jpg' for i in range(1,11)]
    images = [cv2.imread(part_path) for part_path in part_paths]
    dict = get_part_1_info(images, dict)

    part_paths = [f'{seperate_path}/part_2_{i}.jpg' for i in range(1, 9)]
    images = [cv2.imread(part_path) for part_path in part_paths]
    dict = get_part_2_info(images, dict)

    part_paths = [f'{seperate_path}/part_3_{i}.jpg' for i in range(1, 3)]
    images = [cv2.imread(part_path) for part_path in part_paths]
    dict = get_part_3_info(images, dict)

    jsons_path = './data/jsons'
    with open(f'{jsons_path}/{image_name}.json', 'w', encoding = 'utf-8') as f:
        json.dump(dict, f, ensure_ascii = False, indent = 4)

    return "이미지를 업로드 했습니다."

@app.post("/get_part_1_json")
async def get_part_1_json(name:str = Form(...)):
    json_path = f'./data/jsons/{name}.json'
    
    need_keys = ["address", "bun", "ji", "dongName", "floorName", "hoName",
                 "structUse", "landArea", "bldgArea", "exclArea", "deposit",
                 "downPayment", "interimPayment", "balance", "interimDate", "balanceDate"]
    
    with open(f'{json_path}', 'r', encoding = 'utf-8') as f:
        dict = json.load(f)
    
    return {key: dict[key] for key in need_keys if key in dict}

@app.post("/get_part_2_json")
async def get_part_2_json(name:str = Form(...)):
    json_path = f'./data/jsons/{name}.json'
    
    need_keys = ["lessorName", "lessorAddress", "lesseeName", "lesseeAddress",
                 "lessorId", "lesseeId", "lessorPhone", "lesseePhone"]
    
    with open(f'{json_path}', 'r', encoding = 'utf-8') as f:
        dict = json.load(f)
    
    return {key: dict[key] for key in need_keys if key in dict}

@app.post("/get_part_3_json")
async def get_part_3_json(name:str = Form(...)):
    json_path = f'./data/jsons/{name}.json'
    
    need_keys = ["clausesResponse", "scResponse"]
    
    with open(f'{json_path}', 'r', encoding = 'utf-8') as f:
        dict = json.load(f)
    
    return {key: dict[key] for key in need_keys if key in dict}