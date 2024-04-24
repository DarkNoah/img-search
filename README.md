# 图片搜索服务

## 安装环境
```
pip install -r requirements.txt

```
## 运行服务
```
python src/main.py --host "127.0.0.1" --port 8000 --pg_host localhost --pg_port 5432 --pg_user pgvector --pg_password pgvector --pg_dbname postgres
```

访问 http://127.0.0.1:8000/docx

### 创建一个集合
- name 表名称
- mode 必须是 null,"face","child" 之一
- columns key value格式 key为列名称 value为字段类型
- 默认创建 embedding字段,请勿在columns中重复创建,vector长度为模型长度,face模式下固定为512,null和child模式下为2560
- 默认创建 bbox 字段,请勿在columns中重复创建,只有在mode为face, child才有bbox字段内容,记录了检测目标框 <code>[w, y, x+w, y+h]</code>
  
```
# 请求
curl -X 'POST' \
  'http://127.0.0.1:8000/collection/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "name": "test",
  "mode": null,
  "columns": {
    "field1":"varchar(100)",
    "field2": "int8"
  }
}'

# 返回
{
  "name": "test",
  "columns": {
    "id": "int8",
    "embedding": "vector",
    "field2": "int8",
    "bbox": "varchar",
    "field1": "varchar"
  }
}
```

### 删除一个集合
- collection为创建集合时的name

```
curl -X 'DELETE' \
  'http://127.0.0.1:8000/collection/?collection=test' \
  -H 'accept: application/json'
```

### 查询所有集合信息
- 所有表和字段信息
```
# 请求
curl -X 'GET' \
  'http://127.0.0.1:8000/collection/' \
  -H 'accept: application/json'

# 返回
[
  {
    "name": "test",
    "columns": {
      "bbox": "varchar",
      "field1": "varchar",
      "id": "int8",
      "embedding": "vector",
      "field2": "int8"
    }
  }
]
```

### 查询集合中的所有记录
- embedding字段不会返回
```
# 请求
curl -X 'GET' \
  'http://127.0.0.1:8000/collection/test?skip=0&limit=5' \
  -H 'accept: application/json'

# 返回
[
  {
    "id": 1,
    "bbox": null,
    "field1": "abc",
    "field2": 111
  }
]
```



### 存入图片
- 模式为null
```
# 请求
curl -X 'POST' \
  'http://127.0.0.1:8000/collection/test/upload?input=%7B%22field1%22%3A%22abc%22%2C%22field2%22%3A111%7D' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@en.png;type=image/png'

# 返回
[
  {
    "id": 1,
    "bbox": null,
    "field1": "abc",
    "field2": 111
  }
]
```
- 模式为face
```
# 请求
curl -X 'POST' \
  'http://127.0.0.1:8000/collection/face/upload?mode=face' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@97ba2d22521a4aa7f5a520b9a0ceca7.jpg;type=image/jpeg'

# 返回一个或多个人脸
[
  {
    "id": 5,
    "bbox": "[461,201,704,547]"
  },
  {
    "id": 6,
    "bbox": "[461,201,704,547]"
  }
]

```
- 模式为child
```
```


### 图片搜索
模式为null
- filter 存入条件搜索,可为空
- limit 最多根据相似度检索前N条
- file 上传图片
- 返回的distance为相似度 越小越相似
```
# 请求
curl -X 'POST' \
  'http://127.0.0.1:8000/collection/string/search' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'filter=' \
  -F 'limit=5' \
  -F 'file=@e279eb2c1eb1499a9bfa9d1ffef19d64.jpg;type=image/jpeg'

# 返回
[
  {
    "distance": 42.057082849068806,
    "id": 11,
    "bbox": "[58,31,132,92]"
  },
  {
    "distance": 42.916348729607115,
    "id": 10,
    "bbox": "[36,28,109,84]"
  },
  {
    "distance": 45.21452635552746,
    "id": 14,
    "bbox": "[839,577,1405,953]"
  },
  {
    "distance": 45.21452635552746,
    "id": 7,
    "bbox": "[839,577,1405,953]"
  },
  {
    "distance": 46.862409767517825,
    "id": 12,
    "bbox": "[99,78,297,234]"
  }
]
```
模式为face
- 返回的为上传图片中所有人面集合,如果图片中有3个人,那么返回的就是3个人面集合,bbox为人面的位置
- 返回中result为最相似人脸的记录,distance为相似度,bbox为人脸位置,返回的result数量为limit,前最相似的前N(limit)条记录
```
# 请求
curl -X 'POST' \
  'http://127.0.0.1:8000/collection/face/search?mode=face' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'filter=' \
  -F 'limit=5' \
  -F 'file=@97ba2d22521a4aa7f5a520b9a0ceca7.jpg;type=image/jpeg'

# 返回
[
  {
    "index": 0,
    "bbox": "[461,201,704,547]",
    "result": [
      {
        "distance": 0,
        "id": 4,
        "bbox": "[461,201,704,547]"
      },
      {
        "distance": 0,
        "id": 5,
        "bbox": "[461,201,704,547]"
      }
    ]
  }
]
```


