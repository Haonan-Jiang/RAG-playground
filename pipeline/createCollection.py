# 导入所需库
import time
from pymilvus import MilvusClient, DataType


COLLECTION_NAME="point1"

# 网上效果测评：https://zhuanlan.zhihu.com/p/679166797

# 1. 创建Milvus客户端实例，连接到本地Milvus服务
client = MilvusClient(uri="http://localhost:19530")


res = client.drop_collection(
    collection_name=COLLECTION_NAME
)
print(res)

# 准备集合的模式设定
schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

# 添加字段到模式中：


schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="point_id", datatype=DataType.VARCHAR,max_length=32)
schema.add_field(field_name="point_name", datatype=DataType.VARCHAR,max_length=200)
schema.add_field(field_name="point_content", datatype=DataType.VARCHAR,max_length=11000)
schema.add_field(field_name="item_id_cate", datatype=DataType.VARCHAR,max_length=100)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1792)

# 准备索引参数
index_params = client.prepare_index_params()
# 为"id"字段添加STL_SORT类型的索引，适用于整型数据的排序
index_params.add_index(field_name="id", index_type="STL_SORT")
# 为"my_vector"字段添加IVF_FLAT类型的向量索引，使用内积(Inner Product, IP)作为度量方式，nlist参数设置为128
index_params.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128})
#index_params.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="L2", params={"nlist": 128})
#index_params.add_index(field_name="vector", index_type="HNSW", metric_type="L2", params={"M": 8,"efConstruction": 64})

# 使用定义好的模式和索引参数创建集合名为"customized_setup_1"的集合
client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)

# 等待5秒以确保集合创建完成并索引生效
time.sleep(5)

# 获取并打印"customized_setup_1"集合的加载状态
res = client.get_load_state(collection_name=COLLECTION_NAME)
print(res)