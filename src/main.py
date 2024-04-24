import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import os
from pgvector.psycopg import register_vector
import psycopg
from fastapi.responses import JSONResponse
import timm
from PIL import Image, ImageDraw
from io import BytesIO
import torch
from YOLOv8_face import YOLOv8_face
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1
from typing import Dict, Any, Union
from ultralytics import YOLO
import json
from typing import Optional, Union


PG_HOST = os.environ.get("PG_HOST") or "localhost"
PG_PORT = os.environ.get("PG_PORT") or "5432"
PG_USER = os.environ.get("PG_USER") or "pgvector"
PG_PASSWORD = os.environ.get("PG_PASSWORD") or "pgvector"
PG_DEFAULT_DBNAME = os.environ.get("PG_DEFAULT_DBNAME") or "postgres"
MODELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

model = None
transforms = None
face_model = None
img_child_model = None
face_resnet = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device is {device}")


def init_model():
    global transforms
    global model
    global face_model
    global device
    global face_resnet
    global img_child_model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    face_model = YOLOv8_face(
        os.path.join(MODELS_PATH, "yolov8n-face.onnx"), conf_thres=0.7, iou_thres=0.5
    )
    face_resnet = (
        InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(device)
    )
    if os.path.isfile(os.path.join(MODELS_PATH, "img_child", "best.pt")):
        img_child_model = YOLO(os.path.join(MODELS_PATH, "img_child", "best.pt"))

    model = timm.create_model(
        "repvgg_b2.rvgg_in1k",
        pretrained=True,
        num_classes=0,
    )
    model = model.to(device).eval()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)


def insert(table: str, embedding, data: dict):

    fields = ", ".join(data.keys())
    placeholders = ", ".join(["%s"] * len(data))

    if len(data.keys()) > 0:
        fields = ", " + fields
        placeholders = ", " + placeholders
    values = list(data.values())
    values.insert(0, embedding)

    insert_query = f"INSERT INTO {table} (embedding {fields}) VALUES (%s {placeholders}) RETURNING *"
    cur = conn.cursor()
    cur.execute(insert_query, values)
    row = cur.fetchone()
    json_row = {}
    for idx, col in enumerate(cur.description):
        if col.name == "embedding":
            continue
        json_row[col.name] = row[idx]
    conn.commit()
    cur.close()
    return json_row


def query(table: str, embedding, filter: str, limit: int = 5):

    if filter is not None:
        filter = "where " + filter
    else:
        filter = ""
    cur = conn.cursor()
    query = f"select embedding <-> %s AS distance ,* from {table} {filter} ORDER BY distance LIMIT {str(limit)}"
    rows = cur.execute(query, [embedding]).fetchall()
    results = []
    for row in rows:
        json_row = {}
        for idx, col in enumerate(cur.description):
            if col.name == "embedding":
                continue
            json_row[col.name] = row[idx]
        results.append(json_row)
    cur.close()
    return results
    pass


def extract_embedding(img: Image):
    global transforms
    global model
    dd = transforms(img).unsqueeze(0).to(device)
    output = model.forward_features(dd)
    output = model.forward_head(output, pre_logits=True)
    return output[0].detach().cpu().numpy()


def extract_face_embedding(img: Image):
    img = img.resize((160, 160))
    tensor_img = (
        torch.from_numpy(np.array(img)).permute(2, 0, 1).float().to(device) / 255.0
    ).unsqueeze(0)
    face_embedding = face_resnet(tensor_img)
    face_embedding = face_embedding[0].detach().cpu().numpy()
    return face_embedding


conn = psycopg.connect(
    dbname=PG_DEFAULT_DBNAME,
    host=PG_HOST,
    port=PG_PORT,
    user=PG_USER,
    password=PG_PASSWORD,
    autocommit=True,
)

conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(conn)

# conn.execute("DROP TABLE IF EXISTS documents")
# conn.execute(
#     "CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(384))"
# )

app = FastAPI()


class Collection(BaseModel):
    name: str
    mode: Union[str, None]
    columns: dict


class SearchInput(BaseModel):
    filter: str = None


@app.post(
    "/collection/",
    summary="创建集合",
    description="默认创建 embedding, bbox 字段,请勿在columns中重复创建",
)
async def create_collection(input: Collection):
    global model
    try:
        columns = ""
        for key in input.columns:
            value = input.columns[key]
            columns += ", " + key + " " + value
        dis = model.num_features
        if input.mode == "face":
            dis = 512
        conn.execute(
            f"CREATE TABLE {input.name} (id bigserial PRIMARY KEY, embedding vector({str(dis)}), bbox VARCHAR(200) {columns})"
        )
        query = f"SELECT column_name, udt_name FROM information_schema.columns WHERE table_name = '{input.name}'"
        rows = conn.execute(query).fetchall()
        tableInfo = {"name": input.name, "columns": {}}
        for row in rows:
            tableInfo["columns"][row[0]] = row[1]

        return JSONResponse(status_code=200, content=tableInfo)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.delete("/collection/")
async def delete_collection(collection: str):
    try:
        conn.execute(f"DROP TABLE IF EXISTS {collection}")
        return JSONResponse(status_code=200, content=None)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.get(
    "/collection/",
    summary="获取所有集合信息",
)
async def get_collection():
    global model
    try:
        query = f"SELECT table_name,column_name, udt_name FROM information_schema.columns where table_schema = 'public'"
        rows = conn.execute(query).fetchall()
        tables = []
        for row in rows:
            table_name, column_name, udt_name = row
            found = False
            for item in tables:
                if item["name"] == table_name:
                    found = True
                    item["columns"][column_name] = udt_name
                    break
            if not found:
                col = {}
                col[column_name] = udt_name
                tables.append({"name": table_name, "columns": col})

        return JSONResponse(status_code=200, content=tables)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.post(
    "/collection/{collection}/upload",
    summary="上传图片入库",
    description="图片特征提取并入库 mode 只支持 留空, face, child | input 字段为json对应存入的字段信息{'a':1,'b':'xx'}",
)
async def upload(
    collection: str,
    mode: str = None,
    input=None,
    file: UploadFile = File(...),
):
    global face_model
    global face_resnet
    global device
    image_stream = None
    try:
        json_data = {}
        if input is not None:
            json_data = json.loads(input)
        contents = await file.read()
        image_stream = BytesIO(contents)
        img = Image.open(image_stream)

        if img.mode == "RGBA":
            img = img.convert("RGB")
        embedding = None
        if mode is None:
            embedding = extract_embedding(img)
            res = [insert(collection, embedding, json_data)]
            return JSONResponse(status_code=200, content=res)
        elif mode == "face":
            np_img = np.array(img)
            bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            boxes, scores, classids, kpts = face_model.detect(bgr_img)
            res = []
            if boxes is None or len(boxes) == 0:
                return JSONResponse(status_code=500, content={"error": "no faces"})
            else:
                for index, box in enumerate(boxes):
                    if scores[index] < 0.8:
                        continue
                    x, y, w, h = box.astype(int)
                    cropped_image = img.crop((x, y, x + w, y + h))
                    face_embedding = extract_face_embedding(cropped_image)
                    json_data["bbox"] = f"[{str(x)},{str(y)},{str(x+w)},{str(y+h)}]"
                    res.append(insert(collection, face_embedding, json_data))
                    pass

            return JSONResponse(status_code=200, content=res)
        elif mode == "child":
            results = img_child_model(img)
            result = results[0]
            boxes = result.boxes  # Boxes object for bounding box outputs
            res = []
            if len(boxes) == 0:
                embedding = extract_embedding(img)
                res.append(insert(collection, embedding, json_data))
                return JSONResponse(status_code=200, content=res)
            else:
                for box in boxes:
                    xywh = box.xywh[0].cpu().numpy()
                    x, y, w, h = xywh.astype(int)
                    cropped_image = img.crop((x, y, x + w, y + h))
                    embedding = extract_embedding(cropped_image)
                    json_data["bbox"] = f"[{str(x)},{str(y)},{str(x+w)},{str(y+h)}]"
                    res.append(insert(collection, embedding, json_data))
                return JSONResponse(status_code=200, content=res)
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "mode must be [None, face, child]"},
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    finally:
        if image_stream is not None:
            image_stream.close()
        file.close()


@app.delete(
    "/collection/{collection}",
    summary="获取集合记录",
)
async def delete(collection: str, id: str):
    try:
        conn.execute(f"DELETE from {collection} where id = '{id}'")
        return JSONResponse(status_code=200, content=None)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.post(
    "/collection/{collection}/search",
    summary="检索图片",
    description="mode 只支持 留空, face, child",
)
async def search(
    collection: str,
    mode: str = None,
    filter: Optional[str] = Form(None),
    limit: Optional[int] = Form(5),
    file: UploadFile = File(...),
):

    image_stream = None
    try:
        contents = await file.read()
        image_stream = BytesIO(contents)
        img = Image.open(image_stream)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        if img.mode == "RGBA":
            img = img.convert("RGB")
        embedding = None
        if mode is None:
            embedding = extract_embedding(img)
            res = query(collection, embedding, filter, limit)
            return JSONResponse(status_code=200, content=res)
        elif mode == "face":
            np_img = np.array(img)
            bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            boxes, scores, classids, kpts = face_model.detect(bgr_img)
            res = []
            if boxes is None or len(boxes) == 0:
                return JSONResponse(status_code=500, content={"error": "no faces"})
            else:
                for index, box in enumerate(boxes):
                    if scores[index] < 0.8:
                        continue
                    x, y, w, h = box.astype(int)
                    cropped_image = img.crop((x, y, x + w, y + h))
                    face_embedding = extract_face_embedding(cropped_image)
                    q_res = query(collection, face_embedding, filter, limit)
                    bbox = f"[{str(x)},{str(y)},{str(x+w)},{str(y+h)}]"
                    res.append({"index": index, "bbox": bbox, "result": q_res})
                    pass
            pass
            return JSONResponse(status_code=200, content=res)
        elif mode == "child":
            results = img_child_model(img)
            result = results[0]
            boxes = result.boxes  # Boxes object for bounding box outputs
            res = []
            if len(boxes) == 0:
                embedding = extract_embedding(img)
                q_res = query(collection, embedding, filter, limit)
                res.append({"index": 0, "bbox": None, "result": q_res})
            else:

                for index, box in enumerate(boxes):
                    xywh = box.xywh[0].cpu().numpy()
                    x, y, w, h = xywh.astype(int)
                    cropped_image = img.crop((x, y, x + w, y + h))
                    embedding = extract_embedding(cropped_image)
                    q_res = query(collection, embedding, filter, limit)
                    bbox = f"[{str(x)},{str(y)},{str(x+w)},{str(y+h)}]"
                    res.append({"index": index, "bbox": bbox, "result": q_res})
            return JSONResponse(status_code=200, content=res)
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "mode must be [None, face, children]"},
            )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    finally:
        if image_stream is not None:
            image_stream.close()
        file.close()


@app.get("/collection/{collection}")
async def get(
    collection: str,
    filter: Optional[str] = None,
    skip: int = 0,
    limit: Optional[int] = 5,
):
    try:
        if filter is not None:
            filter = "where " + filter
        else:
            filter = ""
        cur = conn.cursor()
        query = f"select * from {collection} {filter} OFFSET {skip} LIMIT {str(limit)}"
        rows = cur.execute(query).fetchall()
        results = []
        for row in rows:
            json_row = {}
            for idx, col in enumerate(cur.description):
                if col.name == "embedding":
                    continue
                json_row[col.name] = row[idx]
            results.append(json_row)
        cur.close()
        return JSONResponse(status_code=200, content=results)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


# # 读取资源
# @app.get("/items/{item_id}")
# async def read_item(item_id: int):
#     global db
#     if item_id not in db:
#         raise HTTPException(status_code=404, detail="Item not found")
#     return db[item_id]


# # 更新资源
# @app.put("/items/{item_id}")
# async def update_item(item_id: int, item: Item):
#     global db
#     if item_id not in db:
#         raise HTTPException(status_code=404, detail="Item not found")
#     db[item_id] = item.dict()
#     return {"message": "Item updated successfully"}


# # 删除资源
# @app.delete("/items/{item_id}")
# async def delete_item(item_id: int):
#     global db
#     if item_id not in db:
#         raise HTTPException(status_code=404, detail="Item not found")
#     del db[item_id]
#     return {"message": "Item deleted successfully"}


if __name__ == "__main__":
    # 配置主机和端口
    init_model()
    # results = img_child_model("302128C47231214010-0013.jpg")
    # for result in results:
    #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     v = len(boxes)
    #     for box in boxes:
    #         print(box.xywh[0].cpu().numpy())

    uvicorn.run(app, host="127.0.0.1", port=8000)
