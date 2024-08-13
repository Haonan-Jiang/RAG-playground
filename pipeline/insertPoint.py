import pandas as pd
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

df = pd.read_excel('citiao.xlsx', engine='openpyxl')
model = SentenceTransformer('lier007/xiaobu-embedding-v2')

data = []
idx = 1
for index, row in df.iterrows():
    # 对每个单元格进行处理
    cell_values = row.to_dict()

    entry = {
        "id": idx,
        "point_id": cell_values.get("point_id"),
        "point_name": cell_values.get("point_name"),
        "point_content": cell_values.get("point_content"),
        "item_id_cate": cell_values.get("item_id_cate"),
        # "vector": model.encode(cell_values.get("point_name")+':'+cell_values.get("point_content")),
        "vector": model.encode(f"{cell_values.get('point_name')}:{cell_values.get('point_content')}")
    }
    idx = idx + 1
    data.append(entry)

# 1. 创建Milvus客户端实例，连接到本地Milvus服务
# print(data.count())
client = MilvusClient(uri="http://localhost:19530")
res = client.insert(
    collection_name="point1",
    data=data
)

print(res)
