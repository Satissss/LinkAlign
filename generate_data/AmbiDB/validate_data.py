from typing import List

import pandas as pd
import json
from utils import parse_schema_from_df


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


if __name__ == "__main__":
    with open(r"new_tables.json", 'r', encoding='utf-8') as file:
        tables = json.load(file)

    for db in tables[:10]:
        db_context = get_schema_context_from_db(db)
        print(db_context)

    # 统计平均单个数据库规模
    size = 0
    for db in tables:
        size += len(db)
    print(f"单个数据库平均字段数量为：{size / len(tables)}")
