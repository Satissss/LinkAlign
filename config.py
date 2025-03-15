# -*- coding: utf-8 -*-
# 本地文件存储目录
ALL_DATABASE_DATA_SOURCE = r"..."

# 训练数据存储目录
DATASET_PATH = r"..."

# 索引保存目录
PERSIST_DIR = ALL_DATABASE_DATA_SOURCE + r"\vector_store"

# 日志目录
LOG_DIR = r"..."

# SummaryIndex 索引文件存储目录
ALL_DATABASE_SUMMARY_PERSIST_DIR = ALL_DATABASE_DATA_SOURCE + r"\vector_store\SummaryIndex"

# VectorStoreIndex 索引文件存储目录
ALL_DATABASE_VECTOR_PERSIST_DIR = ALL_DATABASE_DATA_SOURCE + r"\vector_store\VectorStoreIndex"

# 本地索引文件存储目录
VECTOR_STORE_PERSIST_DIR = r"E:\documents_for_llms\data03\vector_store"

# 文件存储目录索引是否存在。注意：更新文件目录后第一次使用需要设置为 False
IS_VECTOR_STORE_EXIST = True

# 嵌入模型名称
EMBED_MODEL_NAME = None

# 底层大模型名称
LLM_NAME = "zhipu"

# 过程可视化
VERBOSE = False

ZHIPU_API_KEY = "..."
ZHIPU_MODEL = "glm-4-flash"  # 测试哪种模型效果更好

QWEN_API_KEY = "..."
QWEN_MODEL = "qwen-turbo-1101"

DEEPSEEK_API = "..."
DEEPSEEK_MODEL = "deepseek-chat"

# 两个模型的公共参数
TEMPERATURE = 0.45

MAX_OUTPUT_TOKENS = 4096

CONTEXT_WINDOW = 120000

