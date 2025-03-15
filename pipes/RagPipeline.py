import os

from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    SummaryIndex,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
    get_response_synthesizer
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.base import BaseIndex
from llama_index.core.retrievers import VectorIndexRetriever, SummaryIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from prompts.PipelinePromptStore import *

from llms.zhipu.ZhipuModel import ZhipuModel
from embed_model.EmbedModelPathMap import embed_model_map_name_to_path
from typing import Union, List


class RagPipeLines:
    # todo 需要增加一个方法，使用户能够统一修改自定义的默认参数
    DEFAULT_MODEL = ZhipuModel()
    EMBED_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
    ROOT_PERSIST_DIR: str = r"../vector_store"

    COUNT = 1

    @classmethod
    def build_index_from_source(
            cls,
            data_source: Union[str, List[str]],
            persist_dir: str = None,
            is_vector_store_exist: bool = False,
            llm=None,
            index_method: str = None,
            embed_model_name=None,  # 暂时只能支持HuggingFace的嵌入模型
            parser=None,
    ):
        if llm is None:
            llm = cls.DEFAULT_MODEL
        Settings.llm = llm

        if embed_model_name is None:
            embed_model_name = cls.EMBED_MODEL_NAME
        embed_model_name = (embed_model_name if embed_model_name not in embed_model_map_name_to_path.keys() else
                            embed_model_map_name_to_path[embed_model_name])  # 如果预训练模型在本地保存，则优先使用本地模型
        Settings.embed_model = HuggingFaceEmbedding(embed_model_name)

        index_method = None if index_method and index_method not in ["SummaryIndex",
                                                                     "VectorStoreIndex"] else index_method

        if parser is None:
            parser = SentenceSplitter(chunk_size=10000, chunk_overlap=0)

        if is_vector_store_exist:
            if persist_dir is None:
                raise Exception("请保证 vector_store 存在时，persist_dir 非空！")
            # 使用本地持久化存储的vector_store
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            return index

        if persist_dir is None:
            persist_dir = f"{cls.ROOT_PERSIST_DIR}/task{cls.COUNT}_vector_store"

        os.makedirs(persist_dir, exist_ok=True)  # 保证目录已存在

        is_dir = True  # 保存信息源是目录还是文件，如果传入多个文件，则默认当做目录

        if type(data_source) == List:
            is_dir = False
        elif type(data_source) == str:
            from pathlib import Path
            is_dir = Path(data_source).is_dir()

        is_vector_store_method = is_dir

        if index_method:
            is_vector_store_method = True if index_method == "VectorStoreIndex" else False

        if is_vector_store_method:
            # 加载数据并转换为document
            documents = SimpleDirectoryReader(data_source).load_data()

            # 通过 VectorIndex 建立索引
            index = VectorStoreIndex.from_documents(documents, transformations=[parser], show_progress=True)

            # 保存index
            index.storage_context.persist(persist_dir=persist_dir)
        else:
            if type(data_source) == List:
                documents = SimpleDirectoryReader(input_files=data_source).load_data()
            else:
                if not is_dir:
                    documents = SimpleDirectoryReader(input_files=[data_source]).load_data()
                else:
                    documents = SimpleDirectoryReader(data_source).load_data()

            # 通过 VectorIndex 建立索引
            index = SummaryIndex.from_documents(documents, transformations=[parser], show_progress=True)

            # 保存index
            index.storage_context.persist(persist_dir=persist_dir)

        return index

    @classmethod
    def get_query_engine(
            cls,
            index: Union[SummaryIndex, VectorStoreIndex] = None,
            query_template: str = None,
            similarity_top_k=5,
            node_ids: List[str] = None,
            **kwargs
    ):
        if index is None:
            raise Exception("输入 index 不能为空")
        if query_template is not None:
            query_template = PromptTemplate(query_template)
        else:
            query_template = PromptTemplate(DEFAULT_PROMPT_TEMPLATE)

        engine = None
        if type(index) == SummaryIndex:
            engine = index.as_query_engine(text_qa_template=query_template,
                                           similarity_top_k=similarity_top_k,
                                           **kwargs)
        elif type(index) == VectorStoreIndex:
            retriever = VectorIndexRetriever(index=index,
                                             similarity_top_k=similarity_top_k,
                                             node_ids=node_ids,
                                             **kwargs)
            engine = RetrieverQueryEngine.from_args(retriever=retriever,
                                                    text_qa_template=query_template)
        return engine

    @classmethod
    def get_retriever(
            cls,
            index: VectorStoreIndex = None,
            similarity_top_k: int = 5,
            node_ids: List[str] = None,
            **kwargs
    ):
        if index is None:
            raise Exception("输入 index 不能为空")
        retriever = VectorIndexRetriever(index=index,
                                         similarity_top_k=similarity_top_k,
                                         node_ids=node_ids,
                                         **kwargs)
        return retriever


if __name__ == "__main__":
    vector_dir = r"..."
    vector_index = RagPipeLines.build_index_from_source(
        data_source=vector_dir,
        persist_dir=vector_dir + r"\vector_store",
        is_vector_store_exist=True,
        index_method="VectorStoreIndex"
    )
