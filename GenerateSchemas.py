import json
import os

import pandas as pd
from llama_index.core.indices.vector_store import VectorIndexRetriever

from tools.SchemaLinkingTool import SchemaLinkingTool
from utils import get_sql_files, parse_schema_from_df, parse_schemas_from_nodes
from pipes.RagPipeline import RagPipeLines
from llms.qwen.QwenModel import QwenModel
from tqdm import tqdm
import concurrent.futures

llm = QwenModel(model_name="qwen-turbo", temperature=0.85)
filter_llm = QwenModel(model_name="qwen-turbo", temperature=0.42)


def load_db_size(db_id: str):
    db_size = [row["count"] for row in db_info if row["db_id"].lower() == db_id.lower()][0]
    return db_size


def parse_schemas_from_file(db_id: str):
    base_schema_dir = r".\spider2_dev\schemas"
    file_lis = get_sql_files(rf"{base_schema_dir}\{db_id}", ".json")

    schema_lis = []
    for f in file_lis:
        try:
            file_path = rf"{base_schema_dir}\{db_id}\{f}.json"
            with open(file_path, 'r', encoding="utf-8") as file:
                col_info = json.load(file)
            meta_data = col_info["meta_data"]
            schema = {
                "Database name": meta_data["db_id"],
                "Table Name": meta_data["table_name"],
                "Field Name": col_info["column_name"],
                'Type': col_info["column_types"],
                'Description': None if not col_info["column_descriptions"] else col_info["column_descriptions"],
                'Example': None if len(col_info["sample_rows"]) == 0 else col_info["sample_rows"][0],  # 若数据示例不为空，则进行补充
                'turn_n': 0  # 保留或者直接由原始问题检索得到，turn_n = 0
            }
            schema_lis.append(schema)
        except:
            pass

    df = pd.DataFrame(schema_lis)

    return df


def response_filtering(
        data: pd.DataFrame,  # todo 是否可以增加 Nodes 作为参数
        question: str,
        chunk_size: int = 250,
        turn_n: int = 2,
        reserve_df: pd.DataFrame = None
):
    df_list = []
    num_rows = data.shape[0]

    for i in range(0, num_rows, chunk_size):
        df_slice = data.iloc[i:i + chunk_size]
        df_list.append(df_slice)

    sub_data_lis = [reserve_df] if reserve_df is not None else []
    for data in df_list:
        schema_context = parse_schema_from_df(data)
        res = SchemaLinkingTool.locate_with_multi_agent(
            llm=filter_llm,
            query=question,
            context_str=schema_context,
            turn_n=turn_n
        )
        schema_links = res.split("[")[1].split("]")[0].strip()
        schema_links = schema_links.split(",")
        schema_links = [link.strip().replace("`", "").replace('"', "").replace("'", "") for link in schema_links]
        temp_lis = []
        for link in schema_links:
            links_field = link.split(".")
            if len(links_field) <= 4:
                if len(links_field[-2:]) == 2:
                    temp_lis.append(links_field[-2:])
            else:
                temp_lis.append(links_field[2:4])
        for table, field in temp_lis:
            data = data.query(f"not (`Table Name` == '{table}' and `Field Name` == '{field}')")

        sub_data_lis.append(data)
    df = pd.concat(sub_data_lis, axis=0, ignore_index=True)
    df = df.drop_duplicates(subset=['Table Name', 'Field Name'], ignore_index=True)

    return df


def load_retrieval_top_k(db_size):
    if db_size <= 200:
        return 30
    elif db_size <= 500:
        return 40
    elif db_size <= 1000:
        return 50
    elif db_size <= 5000:
        return 60
    else:
        return 70


def load_retrieval_turn_n(db_size):
    if db_size <= 50:
        return 1
    elif db_size <= 200:
        return 2
    elif db_size <= 350:
        return 3
    elif db_size <= 1000:
        return 5
    elif db_size <= 5000:
        return 7
    else:
        return 9


def load_post_retrival_param(db_size):
    """ 返回值：
     1: top_k
     2: turn_n
     """
    if db_size <= 200:
        return 5, 1
    elif db_size <= 500:
        return 10, 1
    elif db_size <= 1000:
        return 15, 1
    elif db_size <= 2000:
        return 15, 2
    else:
        return 20, 1


def transform_name(table_name, col_name):
    prefix = rf"{table_name}_{col_name}"
    prefix = prefix if len(prefix) < 100 else prefix[:100]

    syn_lis = ["(", ")", "%", "/"]
    for syn in syn_lis:
        if syn in prefix:
            prefix = prefix.replace(syn, "_")

    return prefix


def set_retriever(
        retriever: VectorIndexRetriever,
        data: pd.DataFrame,
):
    table_lis, col_lis = list(data["Table Name"]), list(data["Field Name"])
    file_name_lis = []
    for table, col in zip(table_lis, col_lis):
        file_name_lis.append(transform_name(table, col))

    index = retriever.index
    sub_ids = []
    doc_info_dict = index.ref_doc_info
    for key, ref_doc_info in doc_info_dict.items():
        if ref_doc_info.metadata["file_name"] not in file_name_lis:
            sub_ids.extend(ref_doc_info.node_ids)
    retriever.change_node_ids(sub_ids)


def load_data(dataset):
    return pd.read_json(dataset)


def get_files(directory, suffix: str = ".sql"):
    # 获取指定目录下指定后缀名的所有文件名(不带后缀)
    sql_files = [f.split(".")[0].strip() for f in os.listdir(directory) if f.endswith(suffix)]
    return sql_files


def load_external_knowledge(instance_id):
    path = r".\preprocessed_data\spider2_dev\external_knowledge"
    all_ids = get_files(path, ".txt")

    if instance_id in all_ids:
        with open(rf"{path}\{instance_id}.txt", "r", encoding="utf-8") as f:
            external = f.read()
        if len(external) > 50:
            external = "\n####[External Prior Knowledge]:\n" + external + "\n"
            return external

    return None


def get_schema(
        db_id: str,
        question: str,
        instance_id: str,
        reserve_size: int = 90,  # 规模小于 100 则全部保留,
        min_retrival_size: int = 250,  # 最小检索规模，超过该阈值，则启动检索
        filter_chunk_size: int = 250,  # 单个过滤块的大小
        post_retrieval_size: int = 90,  # 最好与 reserve_size 设置一致
        post_retrieval_turn: int = 2,
        reserve_rate: float = 0.6,
):
    """ 检索问题需要的数据库模式 """
    file_name = instance_id + "_agent"
    if os.path.isfile(rf"{save_path}\{file_name}.xlsx"):
        return None

    db_size = load_db_size(db_id)

    # print(db_size)
    if db_size <= reserve_size:
        df = parse_schemas_from_file(db_id)
        df.to_excel(rf"{save_path}\{file_name}.xlsx", index=False)
        return df

    vector_dir = rf"{schema_path}\{db_id}"

    vector_index = RagPipeLines.build_index_from_source(
        data_source=vector_dir,
        persist_dir=vector_dir + r"\vector_store",
        is_vector_store_exist=True,
        index_method="VectorStoreIndex"
    )
    retriever = RagPipeLines.get_retriever(index=vector_index)

    if db_size <= min_retrival_size:
        df = parse_schemas_from_file(db_id)
    else:
        # 进行检索
        similarity_top_k = load_retrieval_top_k(db_size)
        turn_n = load_retrieval_turn_n(db_size)
        retriever.similarity_top_k = similarity_top_k

        nodes_lis = SchemaLinkingTool.retrieve_complete_by_multi_agent_debate(llm=llm, question=question,
                                                                              retriever_lis=[retriever],
                                                                              open_locate=False,
                                                                              output_format="node",
                                                                              # logger=logger,
                                                                              retrieve_turn_n=turn_n
                                                                              )
        df = parse_schemas_from_nodes(nodes_lis)

    # 计算需要保留的模式
    turn_n_lis = df["turn_n"].unique().tolist()
    df_lis = []
    for n in turn_n_lis:
        temp_df = df[df["turn_n"] == n]
        df_reserver_rate = 0.55 * pow(reserve_rate, n)
        if df_reserver_rate <= 0.1:
            continue
        temp_df = temp_df.sample(int(len(temp_df) * df_reserver_rate), random_state=42)
        df_lis.append(temp_df)
    reserve_df = pd.concat(df_lis, axis=0, ignore_index=True)

    for _ in range(post_retrieval_turn):
        # 若 df 字段大于 chunk_size，则增加一次过滤
        if len(df) > post_retrieval_size:
            # 进行过滤
            df = response_filtering(data=df, question=question, chunk_size=filter_chunk_size, reserve_df=reserve_df)

    # 进行后检索，避免过滤掉关键字段
    post_top_k, post_turn_n = load_post_retrival_param(db_size)

    set_retriever(retriever, df)  # 对 retriever 重新进行设置，检索剩余的模式
    retriever.similarity_top_k = post_top_k

    nodes_lis = SchemaLinkingTool.retrieve_complete_by_multi_agent_debate(llm=llm, question=question,
                                                                          retriever_lis=[retriever],
                                                                          open_locate=False,
                                                                          output_format="node",
                                                                          # logger=logger,
                                                                          retrieve_turn_n=post_turn_n
                                                                          )
    sub_df = parse_schemas_from_nodes(nodes_lis)
    df = pd.concat([df, sub_df], axis=0)

    df.to_excel(rf"{save_path}\{file_name}.xlsx", index=False)
    return df


def process_row(index, row, pbar):
    external = load_external_knowledge(row["instance_id"])
    question = row["question"] + external if external else row["question"]
    try:
        get_schema(db_id=row["db_id"],
                   question=question,
                   instance_id=row["instance_id"])
    except Exception as e:
        print(e)

    pbar.update(1)


if __name__ == "__main__":
    save_path = rf".\spider2_dev\instance_schemas"
    schema_path = r".\spider2_dev\schemas"
    DATASET = r".\spider2_dev\spider2_dev_preprocessed.json"

    # 加载保存数据库规模的 json 文档
    with open(r'.\spider2_dev\db_info.json', 'r') as file:
        db_info = json.load(file)

    val_df = load_data(DATASET)

    with tqdm(total=val_df.shape[0]) as pbar:
        inputs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            for index, row in val_df.iterrows():
                inputs.append((index, row, pbar))

            executor.map(lambda args: process_row(*args), inputs)
