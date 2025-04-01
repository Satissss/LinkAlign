import json
from typing import Dict, List

base_dir = r'E:\在校学习\科研\大模型环境下数据查询语言生成通用性的研究\benchmark\关系数据库\Spider'


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


def prepare_raw_data():
    with open(rf'{base_dir}\dev.json', 'r') as file:
        d1 = json.load(file)

    with open(rf'{base_dir}\test.json', 'r') as file:
        d2 = json.load(file)

    with open(rf'{base_dir}\train_spider.json', 'r') as file:
        d3 = json.load(file)

    with open(rf'{base_dir}\train_others.json', 'r') as file:
        d4 = json.load(file)

    data = d1 + d2 + d3 + d4
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
    db_info = []
    for db_lis in tables:
        db_id = db_lis[0]['meta_data']['db_id']
        count = len(db_lis)
        db_info.append({
            'db_id': db_id,
            'count': count
        })
    db_ids = [row['db_id'] for row in db_info]
    print(len(db_ids))
    print(db_ids)
    data = [row for row in data if row['db_id'] in db_ids and len(row['query']) > 250]

    with open("./dataset/raw_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def transform_name(db_list: List):
    db_id = db_list[0]['meta_data']['db_id']
    db_id += "_new"
    for row in db_list:
        row['meta_data']['db_id'] = db_id

    return db_list


# prepare_raw_data()

with open("./dataset/raw_data.json", 'r', encoding='utf-8') as file:
    data = json.load(file)

data_lis = []

for ind, row in enumerate(data):
    data_lis.append({
        'instance_id': f'q_{ind}',
        'question': row['question'],
        'db_id': row['db_id'],
        'query': row['query']
    })

with open("./dataset/raw_data.json", "w", encoding="utf-8") as f:
    json.dump(data_lis, f, ensure_ascii=False, indent=4)
