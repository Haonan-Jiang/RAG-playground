import time
from flask import Response, jsonify, request, stream_with_context
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
import os

COLLECTION_NAME = "point1"

# 加载 .env 文件
load_dotenv()

# 然后你可以像使用环境变量一样使用这些配置
OLLAMA_URI = os.getenv("OLLAMA_URI")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
SENTENCE_TRANSFORMER_MODEL_PATH = os.getenv("SENTENCE_TRANSFORMER_MODEL_PATH")
MILVUS_URI = os.getenv("MILVUS_URI")

llm = Ollama(base_url=f"http://{OLLAMA_URI}", model=OLLAMA_MODEL,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_PATH, device="cuda")
# 1. 创建Milvus客户端实例，连接到本地Milvus服务
client = MilvusClient(uri=MILVUS_URI)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=80, length_function=len,
                                               is_separator_regex=False)

llm_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching docuemnts. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)


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
        limit=2,
        search_params=search_params,
        output_fields=["point_name", "point_content"]
    )
    print(res)
    return res


def format_prompt(context, question):
    """格式化提示模板"""
    template = """
    Answer the question based only on the following context，直接输出原文，不需要编造，不需要分析，不需要解释，不要重复输出:
    {context}
    Question: {question}
    """
    return template.format(context=context, question=question)


def get_llm_response(prompt):
    """调用LLM获取回答"""
    start_time = time.time()
    response = llm.invoke(prompt)
    print(f"LLM响应时间: {time.time() - start_time}秒")
    return response


def v_handler():
    """处理POST /ask 请求的主逻辑"""
    print("Post /ask called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")

    vector = encode_query(query)
    return vector


def ask_post_handler():
    """处理POST /ask 请求的主逻辑"""
    print("Post /ask called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")

    vector = encode_query(query)
    context = search_context(vector)
    formatted_prompt = format_prompt(context, query)

    response = get_llm_response(formatted_prompt)
    response_answer = {"answer": response}
    return response_answer


def ask_post_handler_stream():
    """处理POST /ask 请求的主逻辑"""
    print("Post /ask called")
    query = request.json.get("query")
    print(f"query: {query}")

    vector = encode_query(query)
    context = search_context(vector)
    print("context")
    print(context)
    formatted_prompt = format_prompt(context, query)

    response = llm.stream(formatted_prompt)

    # 设置正确的Content-Type以便浏览器正确处理流式数据
    return response

# run the code
vector = encode_query("有没有关于接线端子相关的知识？")
context = search_context(vector)
formatted_prompt = format_prompt(context, "有没有关于接线端子相关的知识？")
response = get_llm_response(formatted_prompt)