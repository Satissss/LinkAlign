import json
import os
import itertools
import pandas as pd
from utils import parse_schema_from_df, parse_json_from_str
from llms.qwen.QwenModel import QwenModel
from typing import List
from prompt import *
import concurrent.futures

""" 生成 AmbiDB 数据集的全部问题和对应的 SQL 语句 """

llm = QwenModel(model_name="qwen-plus", temperature=0.3, stream=True)


def get_schema_context_from_db(db_schema_info: List):
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

    return schema_context


def extract_db_info(db_name: str):
    db_info_ = [db_ for db_ in db_data if db_[0]['meta_data']['db_id'] == db_name][0]
    return db_info_


if __name__ == "__main__":
    """ 加载 Database 和 现有问题 """
    with open("./dataset/new_tables.json", 'r', encoding='utf-8') as file:
        db_data = json.load(file)
    # 使用多线程获取 db_id 列表
    with concurrent.futures.ThreadPoolExecutor() as executor:
        db_lis = list(executor.map(lambda db: db[0]['meta_data']['db_id'], db_data))

    with open("./dataset/raw_data.json", 'r', encoding='utf-8') as file:
        data = json.load(file)

    """ 随机生成 domain_table 上的查询问题 """
    # 筛选所有 domain_table 数据
    domain_db_lis = [db_ for db_ in db_data if db_[0]['meta_data']['db_id'].endswith("_new")]
    # 使用计数器保证 instance_id 唯一
    ind_counter = itertools.count(start=len(data))
    num_per_db = 2


    def process_domain_db(db_):
        results = []
        for _ in range(num_per_db):
            db_id = db_[0]['meta_data']['db_id']
            schema_context = get_schema_context_from_db(db_)
            input_prompt = generate_data_prompt.format(schema_context=schema_context)
            data_ = llm.complete(input_prompt).text
            new_data = parse_json_from_str(data_)
            results.append({
                'instance_id': f"q_{next(ind_counter)}",
                'question': new_data['question'],
                'db_id': db_id,
                'query': new_data['query']
            })
        return results


    new_generated = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_domain_db, db_) for db_ in domain_db_lis]
        for future in concurrent.futures.as_completed(futures):
            try:
                new_generated.extend(future.result())
            except:
                pass

    # 将新生成的数据加入 data 列表
    data.extend(new_generated)

    new_data_lis = []  # 存储所有新生成的查询样本

    """ 使用 raw_table 作为混淆，生成 raw_table 上的查询问题 """


    def process_raw_confuse(row):
        db_id = row['db_id']
        if db_id.endswith("_new") or f"{db_id}_new" not in db_lis:
            return None
        question = row['question']
        target_database = get_schema_context_from_db(extract_db_info(db_id))
        confuse_database = get_schema_context_from_db(extract_db_info(f"{db_id}_new"))
        input_prompt = question_rewriting_prompt.format(question=question, target_database=target_database,
                                                        confuse_database=confuse_database)
        new_question = llm.complete(input_prompt).text
        return {
            'instance_id': f"q_{next(ind_counter)}",
            'db_id': f"{db_id}_new",
            'question': new_question,
            'query': ''
        }


    # 使用线程池处理 data 中的每一行
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_raw_confuse, data))
    # 过滤掉返回 None 的结果
    new_data_lis.extend([r for r in results if r is not None])

    """ 使用 domain_table 作为混淆，生成 raw_table 上的查询问题 """


    def process_domain_confuse(row):
        db_id = row['db_id']
        if not db_id.endswith("_new") or f"{db_id}_new" not in db_lis:
            return None
        question = row['question']
        target_database = get_schema_context_from_db(extract_db_info(f"{db_id}_new"))
        confuse_database = get_schema_context_from_db(extract_db_info(db_id))
        input_prompt = question_rewriting_prompt.format(question=question, target_database=target_database,
                                                        confuse_database=confuse_database)
        new_question = llm.complete(input_prompt).text
        return {
            'instance_id': f"q_{next(ind_counter)}",
            'db_id': f"{db_id.split('_new')[0]}",
            'question': new_question,
            'query': ''
        }


    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_domain_confuse, data))
    new_data_lis.extend([r for r in results if r is not None])

    """ 根据给定的查询问题和数据库的模式信息生成对应的 SQL 语句 """


    def process_sql(row):
        schema_context = get_schema_context_from_db(extract_db_info(row['db_id']))
        input_prompt = generate_sql_prompt.format(database_schema=schema_context, user_query=row['question'])
        sql = llm.complete(input_prompt).text
        row['query'] = sql
        return row


    with concurrent.futures.ThreadPoolExecutor() as executor:
        new_data_lis = list(executor.map(process_sql, new_data_lis))

    """ 重新根据 SQL 语句校对问题，确保二者完全匹配 """


    def process_alignment(row):
        schema_context = get_schema_context_from_db(extract_db_info(row['db_id']))
        input_prompt = sql_alignment_prompt.format(database_schema=schema_context, question=row['question'],
                                                   query=row['query'])
        new_question = llm.complete(input_prompt).text
        row['question'] = new_question
        return row


    with concurrent.futures.ThreadPoolExecutor() as executor:
        new_data_lis = list(executor.map(process_alignment, new_data_lis))

    # 将生成的新数据加入原有数据
    data.extend(new_data_lis)

    """ 保存全部数据库和查询问题 """
    with open("./dataset/new_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
