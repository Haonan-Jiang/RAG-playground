import time
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('lier007/xiaobu-embedding-v2')
client = MilvusClient(uri="http://localhost:19530")

COLLECTION_NAME = 'point1'


def encode_query(query):
    """对查询进行向量化"""
    start_time = time.time()
    vector = model.encode(query)
    print(f"向量化程序运行时间: {time.time() - start_time}秒")
    return vector


def search_context(vector):
    """根据向量搜索上下文"""
    search_params = {
        "metric_type": "COSINE",
        # "params": {"nprobe": 10},
    }
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=[vector],
        anns_field="vector",
        limit=3,
        search_params=search_params,
        output_fields=["point_name", "point_content"]
    )
    print(res)
    a = res[0]
    b = a[0]

    return res


vector = encode_query("有没有关于接线端子相关的知识？")

res = search_context(vector)
print(res)
