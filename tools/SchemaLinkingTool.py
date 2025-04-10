# -*- coding: utf-8 -*-
import asyncio
from datetime import datetime
from llama_index.core.indices.vector_store import VectorIndexRetriever

from config import *
from llms.zhipu.ZhipuModel import ZhipuModel
from typing import Union, List
from llama_index.core import (
    SummaryIndex,
    VectorStoreIndex,
    Settings,
    QueryBundle,

)
from llama_index.core.indices.utils import default_format_node_batch_fn
from llama_index.core.schema import NodeWithScore, TextNode, MetadataMode
from llama_index.core.base.base_retriever import BaseRetriever
from prompts.PipelinePromptStore import *
from pipes.RagPipeline import RagPipeLines
from prompts.AgentPromptStore import *
from utils import *
import logging


class SchemaLinkingTool:
    @classmethod
    def link_schema_by_rag(
            cls,
            llm: ZhipuModel = None,
            index: Union[SummaryIndex, VectorStoreIndex] = None,
            is_add_example: bool = True,
            question: str = None,
            similarity_top_k: int = 5,
            **kwargs
    ) -> str:
        if not index:
            raise Exception("输入参数中索引不能为空！")

        if not question:
            raise Exception("输入参数中用户查询问题不能为空！")

        llm = llm if llm else ZhipuModel()

        Settings.llm = llm

        few_examples = SCHEMA_LINKING_FEW_EXAMPLES if is_add_example else ""

        query_template = SCHEMA_LINKING_TEMPLATE.format(few_examples=few_examples, question=question)

        engine_args = {
            "index": index,
            "query_template": query_template,
            "similarity_top_k": similarity_top_k,  # todo 文本块的数量可能需要动态确定
            **kwargs
        }

        engine = RagPipeLines.get_query_engine(**engine_args)

        response = engine.query(question).response

        return response

    @classmethod
    def retrieve(
            cls,
            retriever_lis: List[BaseRetriever],
            query_lis: List[Union[str, QueryBundle]]
    ) -> List[NodeWithScore]:
        """ 串行化检索 """
        nodes_lis = []

        for retriever in retriever_lis:
            for query in query_lis:
                nodes = retriever.retrieve(query)
                nodes_lis.extend(nodes)

        nodes_lis.sort(key=lambda x: x.score, reverse=True)

        return nodes_lis

    @classmethod
    def parallel_retrieve(
            cls,
            retriever_lis: List[BaseRetriever],  # 每个 retriever 的不同之处在于建立索引的源文档不同，而非同个数据库
            query_lis: List[Union[str, QueryBundle]]  #
    ) -> List[NodeWithScore]:
        async def retrieve_from_single_retriever(retriever: BaseRetriever, query: Union[str, QueryBundle]):
            nodes = await retriever.aretrieve(query)
            return nodes

        # 初始化事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # 为每个检索器创建任务，并在事件循环中运行它们
            tasks = ([loop.create_task(retrieve_from_single_retriever(retriever, query))
                      for query in query_lis
                      for retriever in retriever_lis])
            # 使用 loop.run_until_complete 函数协调所有任务
            results = loop.run_until_complete(asyncio.gather(*tasks))
        finally:
            # 确保事件循环在完成后关闭
            loop.close()

        # 扁平化结果列表
        nodes_lis = [node for sublist in results for node in sublist]

        # 排序结果
        nodes_lis.sort(key=lambda x: x.score, reverse=True)

        return nodes_lis

    @classmethod
    def reason_enhance(
            cls,
            llm=None,
            query: str = None
    ):
        """ 利用大模型在问题的基础上进行推理，并返回推理分析的结果 """
        if not query:
            raise Exception("输入的查询不能为空！")

        llm = llm if llm else ZhipuModel()

        prompt = REASON_ENHANCE_TEMPLATE.format(query=query)

        reason_query = llm.complete(prompt=prompt).text  # 增强后的问题查询

        return reason_query

    @classmethod
    def retrieve_complete(
            cls,
            question: str = None,
            retriever_lis: List[VectorIndexRetriever] = None,
            llm=None,
            open_reason_enhance: bool = True,
            open_locate: bool = False,  # 测试一般设置为关闭，正式实验可以开启
            open_agent_debate: bool = False,  # 只有在open_locate 为真时该参数生效
            turn_n: int = 2,
            output_format: str = "database",  # database or schema,前者输出数据库名称，后者输出该数据库的 schema 信息
            remove_duplicate: bool = True,  # 在已检索出 node 以外的范围检索，效率可能有损失
            is_all: bool = True,
            enhanced_question=None
    ):
        """ 
            Step one: retrieve potential database schemas. 
            Mode: Pipeline. 
        """
        if not question:
            raise Exception("输入参数中问题不能为空！")
        elif not retriever_lis:
            raise Exception("输入参数中索引列表不能为空！")

        llm = llm if llm else ZhipuModel()

        if not open_reason_enhance:
            """ 如果不进行推理增强，仅使用 LlamaIndex 提供的检索功能"""
            nodes = cls.parallel_retrieve(retriever_lis, [question])

        else:
            if not remove_duplicate:
                """ 如果不进行去重，同时使用 Question 和增强后的问题进行检索 """
                analysis = cls.reason_enhance(llm=llm, query=question)  # 调用大模型，通过推理对原始问题进行增强

                enhanced_question = question + analysis

                nodes = cls.parallel_retrieve(retriever_lis, [question, enhanced_question])

            else:
                # 获取所有的 index 和 id 列表
                index_lis = [ret.index for ret in retriever_lis]

                question_nodes = cls.parallel_retrieve(retriever_lis, [question])

                sub_ids = get_sub_ids(question_nodes, index_lis, is_all=is_all)

                # 设置新的id
                for ret in retriever_lis:
                    ret.change_node_ids(sub_ids)

                if enhanced_question is None:
                    # 进行问题增强
                    analysis = cls.reason_enhance(llm=llm, query=question)  # 调用大模型，通过推理对原始问题进行增强
                    enhanced_question = question + analysis

                enhance_question_nodes = cls.parallel_retrieve(retriever_lis, [enhanced_question])

                for ret in retriever_lis:
                    ret.back_to_original_ids()

                nodes = question_nodes + enhance_question_nodes

        nodes.sort(key=lambda node: node.score, reverse=True)
        # print(1)
        if open_locate:
            """ 若进行数据库定位 """
            if open_agent_debate:
                predict_database = cls.locate_with_multi_agent(llm=llm, query=question, nodes=nodes, turn_n=turn_n)
            else:
                schemas = get_all_schemas_from_schema_text(nodes=nodes, output_format='schema')
                predict_database = cls.locate(llm=llm, query=question, context=schemas)

            return predict_database

        else:
            output = get_all_schemas_from_schema_text(nodes=nodes, output_format=output_format,
                                                      schemas_format="str", is_all=is_all)

            return output

    @classmethod
    def retrieve_complete_by_multi_agent_debate(
            cls,
            question: str = None,
            retrieve_turn_n: int = 2,
            locate_turn_n: int = 2,
            retriever_lis: List[VectorIndexRetriever] = None,
            llm=None,
            open_locate: bool = False,  # 测试一般设置为关闭，正式实验可以开启,locate 只能输出唯一的 database
            open_agent_debate: bool = False,
            output_format: str = "database",  # database or schema,前者输出数据库名称，后者输出该数据库的 schema 信息
            remove_duplicate: bool = True,  # 在已检索出 node 以外的范围检索，效率可能有损失
            is_all: bool = True,
            logger=None
    ):
        """
            Step one: retrieve potential database schemas. 
            Mode: Agent
        """
        if logger is None:
            logger = Logger()

        if not question:
            raise Exception("输入参数中问题不能为空！")
        elif not retriever_lis:
            raise Exception("输入参数中索引列表不能为空！")

        llm = llm if llm else ZhipuModel()

        enhanced_question = question
        question_nodes = cls.parallel_retrieve(retriever_lis, [question])
        question_nodes = [set_node_turn_n(node, 0) for node in question_nodes]

        nodes = question_nodes
        temp_tuple_lis = []
        # 获取所有的 index 和 id 列表
        index_lis = [ret.index for ret in retriever_lis]

        sub_ids = get_ids_from_source(nodes)

        for ind in range(retrieve_turn_n):
            logger.info(f"[Query Rewriting Turn {ind + 1}]")
            if not remove_duplicate:
                # 这里没有将检索范围设置为 sub_ids
                nodes += cls.parallel_retrieve(retriever_lis, [enhanced_question])

            else:
                # 设置新的id
                for ret in retriever_lis:
                    ret.change_node_ids(sub_ids)

                if ind != 0:
                    enhance_question_nodes = cls.parallel_retrieve(retriever_lis, [enhanced_question])
                    enhance_question_nodes = [set_node_turn_n(node, ind) for node in enhance_question_nodes]
                    nodes += enhance_question_nodes

                sub_ids = get_sub_ids(nodes, index_lis, is_all)

                # 恢复原来的 id
                for ret in retriever_lis:
                    ret.back_to_original_ids()

            # schemas = get_all_schemas_from_schema_text(nodes=nodes, output_format='schema', is_all=is_all)
            schemas = parse_schema_from_df(parse_schemas_from_nodes(nodes))
            """ 
            下面使用 multi-agent debate 的方式进行，共有两个角色，judge 和 annotator。
            judge 负责分析错误并给出分析，而 annotator 主要负责为问题添加注释
            """
            # judge 进行分析
            analysis = llm.complete(JUDGE_TEMPLATE.format(question=question, context=schemas)).text

            logger.info("[analysis]" + analysis)
            # annotator 添加注释
            annotation = llm.complete(ANNOTATOR_TEMPLATE.format(question=question, analysis=analysis)).text
            logger.info("[annotation]" + annotation)
            enhanced_question = question + annotation

        if not remove_duplicate:
            nodes += cls.parallel_retrieve(retriever_lis, [enhanced_question])
        else:
            # 设置新的id
            for ret in retriever_lis:
                ret.change_node_ids(sub_ids)

            enhance_question_nodes = cls.parallel_retrieve(retriever_lis, [enhanced_question])
            enhance_question_nodes = [set_node_turn_n(node, retrieve_turn_n) for node in enhance_question_nodes]
            nodes += enhance_question_nodes

            # 恢复原来的 id
            for ret in retriever_lis:
                ret.back_to_original_ids()

        # 根据 turn_n 和 score 重新排序, 重写次数[1] 和 分数[2]
        nodes.sort(key=lambda x: (x.metadata["turn_n"], x.score))

        if open_locate:
            """ 若进行数据库定位 """
            if open_agent_debate:
                predict_database = cls.locate_with_multi_agent(llm=llm, query=question, nodes=nodes,
                                                               turn_n=locate_turn_n)
            else:
                schemas = get_all_schemas_from_schema_text(nodes=nodes, output_format='schema', is_all=is_all)
                predict_database = cls.locate(llm=llm, query=question, context=schemas)

            return predict_database

        else:
            output = get_all_schemas_from_schema_text(nodes=nodes, output_format=output_format, is_all=is_all)

            return output

    @classmethod
    def locate(
            cls,
            llm=None,
            query: str = None,
            context: str = None  # 检索的所有数据库schema
    ) -> str:
        """
            Step two: isolate irrelevant schema information.
            Mode: Pipeline
        """
        if not query:
            raise Exception("输入的查询不能为空！")

        llm = llm if llm else ZhipuModel()

        prompt = LOCATE_TEMPLATE.format(query=query, context=context)

        # print(prompt)
        database = llm.complete(prompt=prompt).text  # 增强后的问题查询
        #
        return database

    @classmethod
    def locate_with_multi_agent(
            cls,
            llm=None,
            turn_n: int = 2,
            query: str = None,
            nodes: List[NodeWithScore] = None,
            context_lis: List[str] = None,
            context_str: str = None

    ) -> str:
        """
            Step two: isolate irrelevant schema information.
            Mode: Agent
        """
        if not query:
            raise Exception("输入的查询不能为空！")

        llm = llm if llm else ZhipuModel()

        if context_str or context_lis:
            pass
        elif nodes:
            context_lis = get_all_schemas_from_schema_text(nodes, output_format="schema", schemas_format="list")
        else:
            raise Exception("输入参数中没有包含 database schemas")

        if not context_str:
            context_str = ""
            for ind, context in enumerate(context_lis):
                context_str += f"""
    [The Start of Candidate Database"{ind + 1}"'s Schema]
    {context}
    [The End of Candidate Database"{ind + 1}"'s Schema]
                    """
        source_text = SOURCE_TEXT_TEMPLATE.format(query=query, context_str=context_str)

        chat_history = []

        # one-by-one
        for i in range(turn_n):
            data_analyst_prompt = FAIR_EVAL_DEBATE_TEMPLATE.format(
                source_text=source_text,
                chat_history="\n".join(chat_history),
                role_description=DATA_ANALYST_ROLE_DESCRIPTION,
                agent_name="data analyst"
            )
            data_analyst_debate = llm.complete(data_analyst_prompt).text
            chat_history.append(f"""
    [Debate Turn: {i + 1}, Agent Name:"data analyst", Debate Content:{data_analyst_debate}]
                """)

            data_scientist_prompt = FAIR_EVAL_DEBATE_TEMPLATE.format(
                source_text=source_text,
                chat_history="\n".join(chat_history),
                role_description=DATABASE_SCIENTIST_ROLE_DESCRIPTION,
                agent_name="database scientist"
            )
            data_scientist_debate = llm.complete(data_scientist_prompt).text
            chat_history.append(f"""
    [Debate Turn: {i + 1}, Agent Name:"database scientist", Debate Content:{data_scientist_debate}]
                """)

        # print(chat_history)
        summary_prompt = FAIR_EVAL_DEBATE_TEMPLATE.format(
            source_text=source_text,
            chat_history="\n".join(chat_history),
            role_description=SUMMARY_TEMPLATE,
            agent_name="debate terminator"
        )

        database = llm.complete(summary_prompt).text

        return database

    @classmethod
    def generate_schema(
            cls,
            llm=None,
            query: str = None,
            database: str = None,
            context: str = None,
            logger=None
    ):
        """
            Step there: extract schemas for SQL generation.
            Mode: Pipeline.
        """
        llm = llm if llm else ZhipuModel()
        if not logger:
            logger = logging.getLogger(__name__)

        if context is None:
            with open(ALL_DATABASE_DATA_SOURCE + rf"\{database.lower()}.sql", "r", encoding="utf-8") as file:
                context = file.read().strip()
        context_str = f"[The Start of Database Schemas]\n{context}\n[The End of Database Schemas]"
        query = SCHEMA_LINKING_MANUAL_TEMPLATE.format(few_examples=SCHEMA_LINKING_FEW_EXAMPLES,
                                                      context_str=context_str,
                                                      question=query)
        predict_schema = llm.complete(query).text
        logger.info("[schema linking result]" + "\n" + predict_schema)

        return predict_schema

    @classmethod
    def generate_by_multi_agent(
            cls,
            llm=None,
            query: str = None,
            database: str = None,
            context: str = None,
            turn_n: int = 2,
            linker_num: int = 1,  # schema linker 角色的数量
            logger=None
    ):
        """
            Step there: extract schemas for SQL generation.
            Mode: Agent
        """
        llm = llm if llm else ZhipuModel()
        if not logger:
            logger = logging.getLogger(__name__)

        if context is None:
            with open(ALL_DATABASE_DATA_SOURCE + rf"\{database.lower()}.sql", "r", encoding="utf-8") as file:
                context = file.read().strip()
        context_str = f"""
[The Start of Database Schemas]
{context}
[The End of Database Schemas]
"""
        source_text = GENERATE_SOURCE_TEXT_TEMPLATE.format(query=query, context_str=context_str)

        chat_history = []

        # one-by-one
        for i in range(turn_n):
            data_analyst_prompt = GENERATE_FAIR_EVAL_DEBATE_TEMPLATE.format(
                source_text=source_text,
                chat_history="\n".join(chat_history),
                role_description=GENERATE_DATA_ANALYST_ROLE_DESCRIPTION,
                agent_name="data analyst"
            )
            for j in range(linker_num):
                data_analyst_debate = llm.complete(data_analyst_prompt).text
                chat_history.append(f"""
[Debate Turn: {i + 1}, Agent Name:"data analyst {j}", Debate Content:{data_analyst_debate}]
""")
            data_scientist_prompt = GENERATE_FAIR_EVAL_DEBATE_TEMPLATE.format(
                source_text=source_text,
                chat_history="\n".join(chat_history),
                role_description=GENERATE_DATABASE_SCIENTIST_ROLE_DESCRIPTION,
                agent_name="data scientist"
            )
            data_scientist_debate = llm.complete(data_scientist_prompt).text
            chat_history.append(f"""
[Debate Turn: {i + 1}, Agent Name:"data scientist", Debate Content:{data_scientist_debate}]
""")
        logger.info("[multi-agent debate chat history]" + "\n".join(chat_history))
        summary_prompt = GENERATE_FAIR_EVAL_DEBATE_TEMPLATE.format(
            source_text=source_text,
            chat_history="\n".join(chat_history),
            role_description=GENERATE_SUMMARY_TEMPLATE,
            agent_name="debate terminator"
        )
        schema = llm.complete(summary_prompt).text

        logger.info("[debate summary as schema linking result]" + "\n".join(schema))

        return schema

    @classmethod
    def schema_linking(
            cls,
            question: str = None,
            llm=None,
            turn_n: int = 2,
            retriever_lis: List[VectorIndexRetriever] = None,
            remove_duplicate: bool = True,
            retrieval_mode: str = "agent",  # 两种检索方式，pipeline 或者 agent
            open_agent_debate: bool = False,
            generate_mode: str = "agent"
    ) -> str:
        if not question:
            raise Exception("输入参数中问题不能为空！")
        elif not retriever_lis:
            raise Exception("输入参数中索引列表不能为空！")

        if retrieval_mode not in ["pipeline", "agent"]:
            raise Exception("输入参数中检索模式不正确！")

        llm = llm if llm else ZhipuModel()

        if retrieval_mode == "pipeline":
            database = SchemaLinkingTool.retrieve_complete(question, retriever_lis, llm, open_locate=True,
                                                           remove_duplicate=remove_duplicate,
                                                           open_agent_debate=open_agent_debate)
        else:
            database = SchemaLinkingTool.retrieve_complete_by_multi_agent_debate(question,
                                                                                 turn_n, turn_n,
                                                                                 retriever_lis, llm,
                                                                                 open_locate=True,
                                                                                 remove_duplicate=remove_duplicate,
                                                                                 open_agent_debate=open_agent_debate
                                                                                 )
        with open(ALL_DATABASE_DATA_SOURCE + rf"\{database.lower()}.sql", "r", encoding="utf-8") as file:
            schema = file.read().strip()

        if generate_mode == "agent":
            predict_schema = cls.generate_by_multi_agent(llm=llm, query=question, context=schema, turn_n=turn_n)
        else:
            predict_schema = cls.generate_schema(llm=llm, query=question, context=schema)

        return predict_schema


def filter_nodes_by_database(
        nodes: List[NodeWithScore],
        database: Union[str, List],
        output_format: str = "str"
):
    schema_lis = []
    for node in nodes:
        file_path = node.node.metadata["file_path"]
        db = file_path.split("\\")[-1].split(".")[0].strip()
        if type(database) == str:
            if db == database:
                schema_lis.append(default_format_node_batch_fn([node.node]))
        elif type(database) == List:
            if db in database:
                schema_lis.append(default_format_node_batch_fn([node.node]))
    if output_format == "str":
        return "\n".join(schema_lis)

    return schema_lis


def get_all_schemas_from_schema_text(
        nodes: List[NodeWithScore],
        output_format: str = "database",  # database or schema or node
        schemas_format: str = "str",  # 当输出格式为 node 时无效
        is_all: bool = True
):
    if output_format == "node":
        return nodes

    databases = []

    for node in nodes:
        file_path = node.node.metadata["file_path"]
        db = file_path.split("\\")[-1].split(".")[0].strip()
        databases.append(db)

    databases = list(set(databases))

    if output_format == "database":
        return databases

    if is_all:
        schemas = []
        for path in [node.node.metadata["file_path"] for node in nodes]:
            with open(path, "r", encoding="utf-8") as file:
                schema = file.read().strip()
                schemas.append(schema)

        if schemas_format == "str":
            schemas = "\n".join(schemas)
    else:
        summary_nodes = nodes
        fmt_node_txts = []
        for idx in range(len(summary_nodes)):
            file_path = summary_nodes[idx].node.metadata["file_path"]
            db = file_path.split("\\")[-1].split(".")[0].strip()
            fmt_node_txts.append(
                f"### Database Name: {db}\n#Following is the table creation statement for the database {db}\n"
                f"{summary_nodes[idx].get_content(metadata_mode=MetadataMode.LLM)}"
            )
        schemas = "\n\n".join(fmt_node_txts)

    if output_format == "all":
        return databases, schemas, nodes
    else:
        return schemas


def get_sub_ids(
        nodes: List[NodeWithScore],
        index_lis: List[VectorStoreIndex],
        is_all: bool = True
):
    if is_all:
        file_name_lis = []
        for node in nodes:
            file_name = node.node.metadata["file_name"]
            file_name_lis.append(file_name)

        sub_ids = []
        duplicate_ids = []
        for index in index_lis:
            doc_info_dict = index.ref_doc_info
            for key, ref_doc_info in doc_info_dict.items():
                if ref_doc_info.metadata["file_name"] not in file_name_lis:
                    sub_ids.extend(ref_doc_info.node_ids)
                else:
                    duplicate_ids.extend(ref_doc_info.node_ids)

        return sub_ids
    else:
        exist_node_ids = [node.node.id_ for node in nodes]
        all_ids = []
        for index in index_lis:
            doc_info_dict = index.ref_doc_info
            for key, ref_doc_info in doc_info_dict.items():
                all_ids.extend(ref_doc_info.node_ids)
        sub_ids = [id_ for id_ in all_ids if id_ not in exist_node_ids]

        return sub_ids


def get_ids_from_source(
        source: Union[List[VectorStoreIndex], List[NodeWithScore]]
):
    node_ids = []
    """ 本方法仅用于解析本实验需要两种类型的 node_id """
    for data in source:
        if isinstance(data, VectorStoreIndex):
            doc_info_dict = data.ref_doc_info
            for key, ref_doc_info in doc_info_dict.items():
                node_ids.extend(ref_doc_info.node_ids)

        elif isinstance(data, NodeWithScore):

            node_ids.append(data.node.node_id)

    # 去重
    node_ids = list(set(node_ids))

    return node_ids
