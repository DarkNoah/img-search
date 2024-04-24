import argparse
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

MODELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

model = None
transforms = None
face_model = None
img_child_model = None
face_resnet = None
conn = None
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


def connect_db(PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DEFAULT_DBNAME):
    global conn
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


def insert(table: str, embedding, data: dict):
    global conn
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
    global conn
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


app = FastAPI()


class Collection(BaseModel):
    name: str
    mode: Union[str, None]
    columns: dict


class SearchInput(BaseModel):
    filter: str = None


@app.post(
    "/collection/",
    summary="create collection",
    description="Default creation of [embedding, bbox] fields, Please do not recreate duplicates in columns.",
)
async def create_collection(input: Collection):
    global model
    global conn
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


@app.delete(
    "/collection/",
    summary="delete collection",
)
async def delete_collection(collection: str):
    global conn
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
    summary="get all collection",
)
async def get_collection():
    global model
    global conn
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
    summary="upload",
    description="input 字段为json对应存入的字段信息{'a':1,'b':'xx'}",
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
    summary="delete row",
)
async def delete(collection: str, id: int):
    global conn
    try:
        conn.execute(f"DELETE from {collection} where id = {id}")
        return JSONResponse(status_code=200, content=None)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.post(
    "/collection/{collection}/search",
    summary="search image",
    description="mode only null, face, child",
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
    global conn
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--pg_host", type=str, default=(os.environ.get("PG_HOST", "localhost"))
    )
    parser.add_argument(
        "--pg_port", type=int, default=(int(os.environ.get("PG_PORT", 5432)))
    )
    parser.add_argument(
        "--pg_user", type=str, default=os.environ.get("PG_USER", "pgvector")
    )
    parser.add_argument(
        "--pg_password", type=str, default=os.environ.get("PG_PASSWORD", "pgvector")
    )
    parser.add_argument(
        "--pg_dbname", type=str, default=os.environ.get("PG_DEFAULT_DBNAME", "postgres")
    )

    args = parser.parse_args()

    PG_HOST = args.pg_host
    PG_PORT = args.pg_port
    PG_USER = args.pg_user
    PG_PASSWORD = args.pg_password
    PG_DEFAULT_DBNAME = args.pg_dbname
    connect_db(PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DEFAULT_DBNAME)
    init_model()
    uvicorn.run(app, host=args.host, port=args.port)
