""" 构建 AmbiDB 数据集以适用于大规模、多数据库场景下的模式链接组件的快速设计和测试 """
import json
import os
from typing import Dict, List
import pandas as pd
from llms.zhipu.ZhipuModel import ZhipuModel
from llms.qwen.QwenModel import QwenModel
from llms.yantronic.YanModel import YanModel
from utils import *
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from prompt import *

os.environ['HTTP_PROXY'] = 'socks5://127.0.0.1:10808'
os.environ['HTTPS_PROXY'] = 'socks5://127.0.0.1:10808'

spider_source_dir = r'E:\在校学习\科研\大模型环境下数据查询语言生成通用性的研究\benchmark\关系数据库\Spider'
llm = QwenModel(model_name="qwen-turbo", temperature=0.45, stream=True)


def process_data(row: Dict, exclude_db: List = None):
    """ 将经过 preprocess 的单个数据库拆分为若干 col json 文件"""
    column_info_lis = []
    db_id = row["db_id"]
    if exclude_db is not None:
        if db_id in exclude_db:
            return

    tables = row["table_names_original"]
    columns = row["column_names_original"]
    types = row["column_types"]

    for ind, (table_ind, col_name) in enumerate(columns):
        if col_name == "*":
            continue
        col_info = dict()
        col_info["column_name"] = col_name
        col_info["column_types"] = types[ind]

        table_name = tables[table_ind]

        meta_data = {
            "db_id": db_id,
            "table_name": table_name,

        }
        col_info["meta_data"] = meta_data

        column_info_lis.append(col_info)

    return column_info_lis


def process_table(row):
    db_schema_info = process_data(row)
    schema_lis = []

    for col_info in db_schema_info:
        meta_data = col_info.get("meta_data", {})
        schema = {
            "Database name": meta_data.get("db_id"),
            "Table Name": meta_data.get("table_name"),
            "Field Name": col_info.get("column_name"),
            'Type': col_info.get("column_types"),
            'Description': col_info.get("column_descriptions"),
            'Example': col_info.get("sample_rows")
        }
        schema_lis.append(schema)

    df_schema = pd.DataFrame(schema_lis)
    schema_context = parse_schema_from_df(df_schema)
    domain_schema = llm.complete(extract_domain_schema_prompt.format(schema_context=schema_context)).text
    return parse_json_from_str(domain_schema)


def prepare_data():
    with open(fr'{spider_source_dir}\test_tables.json', 'r') as file:
        test_tables = json.load(file)

    db_info = []
    for row in test_tables:
        db_info.append({
            'db_id': row['db_id'],
            'count': len(row['column_names_original']) - 1
        })
    db_info.sort(key=lambda x: x['count'])
    print(db_info)
    # 对数据库进行筛选，排除一部分模式数量较少的数据库
    min_db_size = 25
    db_info = [row for row in db_info if 100 > row['count'] >= min_db_size]
    db_id_lis = [row['db_id'] for row in db_info]
    test_tables = [row for row in test_tables if row['db_id'] in db_id_lis]

    with open('./dataset/tables.json', 'w', encoding='utf-8') as f:
        json.dump(test_tables, f, ensure_ascii=False, indent=4)

    """ 提取每一个数据库的领域内模式 """
    domain_tables = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(process_table, test_tables)

    domain_tables.extend(results)
    # 保存领域内数据库模式
    with open('./dataset/domain_tables.json', 'w', encoding='utf-8') as f:
        json.dump(domain_tables, f, ensure_ascii=False, indent=4)


def expand_database(db_list: List):
    """ 输入单个数据库的全部模式元数据列表。每个元数据是一个字典 """
    schema_lis = []

    for col_info in db_list:
        meta_data = col_info.get("meta_data", {})
        schema = {
            "Database name": meta_data.get("db_id"),
            "Table Name": meta_data.get("table_name"),
            "Field Name": col_info.get("column_name"),
            'Type': col_info.get("column_types"),
            'Description': col_info.get("column_descriptions"),
            'Example': col_info.get("sample_rows")
        }
        schema_lis.append(schema)
    df_schema = pd.DataFrame(schema_lis)
    schema_context = parse_schema_from_df(df_schema)
    expand_schema = llm.complete(expand_database_schema.format(schema_context=schema_context)).text
    expand_schema = parse_json_from_str(expand_schema)
    db_list += expand_schema

    return db_list


def transform_name(db_list: List):
    db_id = db_list[0]['meta_data']['db_id']
    db_id += "_new"
    for row in db_list:
        row['meta_data']['db_id'] = db_id

    return db_list


if __name__ == "__main__":
    """ Databases """
    """ 准备算法开始的 Database """
    with open('./dataset/tables.json', 'r') as file:
        test_tables = json.load(file)
    test_table_lis = []
    for row in test_tables:
        db_schema_info = process_data(row)
        test_table_lis.append(db_schema_info)

    with open('./dataset/domain_tables.json', 'r') as file:
        domain_tables = json.load(file)

    for db_lis in domain_tables:
        transform_name(db_lis)

    tables = test_table_lis + domain_tables
    expand_turn = 2

    """ 对现有数据库和提取出的领域内数据库进行扩展 """


    def process_db(db_lis, expand_turn):
        for _ in range(expand_turn):
            try:
                expand_database(db_lis)
            except Exception as e:
                print(e)


    def multi_thread_expand(tables, expand_turn, max_workers=None):
        with ThreadPoolExecutor(max_workers=3) as executor:
            # 提交所有任务到线程池
            futures = [executor.submit(process_db, db_lis, expand_turn) for db_lis in tables]

            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                future.result()  # 可以在这里处理异常或结果


    multi_thread_expand(tables, expand_turn)
    # 使用示例
    multi_thread_expand(tables, expand_turn)

    with open("./dataset/new_tables.json", "w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False, indent=4)

    db_info = []
    for db_lis in tables:
        db_id = db_lis[0]['meta_data']['db_id']
        count = len(db_lis)
        db_info.append({
            'db_id': db_id,
            'count': count
        })

