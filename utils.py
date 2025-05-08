import warnings
from os import PathLike
from pathlib import Path
from typing import List, Dict, Union
import json

import pandas as pd
from llama_index.core.schema import NodeWithScore


def parse_list_from_str(string: str = None) -> List[str]:
    """
    Parse a list from string. Expected format: "['a','b','c']"
    """
    try:
        cleaned = string.translate(str.maketrans('', '', '"\'[] \n`'))
        cleaned = cleaned.replace("python", "")

        return cleaned.split(',') if cleaned else []
    except Exception as e:
        print("Error parsing list from string!")
        raise ValueError("Invalid string format for list parsing") from e


def parse_json_from_str(string: str = None) -> dict:
    try:
        cleaned = string.translate(str.maketrans('', '', '\n`'))
        cleaned = cleaned.replace("json", "") if "json" in cleaned else cleaned
        return json.loads(cleaned)
    except Exception as e:
        raise ValueError("Failed to parse JSON from string") from e


def get_sql_files(directory: str, suffix: str = ".sql"):
    return [f.stem for f in Path(directory).iterdir() if f.is_file() and f.suffix == suffix]


def get_all_directories(directory: str):
    return [f.name for f in Path(directory).iterdir() if f.is_dir()]


def parse_schemas_from_nodes(
        nodes: List[NodeWithScore],
        schema_source: Union[str, PathLike] = None,
        output_format: str = "dataframe"
):
    all_schema = []
    for node in nodes:
        if schema_source:
            schema_source = Path(schema_source) if isinstance(schema_source, str) else schema_source
            file_path = schema_source / node.node.metadata["file_name"]
        else:
            file_path = Path(node.node.metadata["file_path"])
        if not file_path.exists():
            warnings.warn(f"读取文件时，给定路径无效，该文件不存在。文件路径为：{file_path}", category=UserWarning)
            continue
        col_info = load_dataset(file_path)
        if not isinstance(col_info, dict):
            continue
        meta_data = col_info["meta_data"]
        schema = {
            "Database name": meta_data["db_id"],
            "Table Name": meta_data["table_name"],
            "Field Name": col_info["column_name"],
            'Type': col_info["column_types"],
            'column_descriptions': col_info.get("column_descriptions", None),
            'sample_rows': col_info.get("sample_rows", None),
            'turn_n': node.metadata.get("turn_n", None)
        }
        all_schema.append(schema)

    output_format = "dataframe" if not output_format else output_format
    if output_format == "dataframe":
        all_schema = pd.DataFrame(all_schema)

    return all_schema


def parse_schema_from_df(df: pd.DataFrame) -> str:
    grouped = df.groupby('table_name')
    output_lines = []

    for table_name, group in grouped:
        columns = []
        for _, row in group.iterrows():
            col_type = row["column_types"]
            if isinstance(col_type, str) and len(col_type) > 150:
                col_type = col_type[:150]
            columns.append(f'{row["column_name"]}(Type: {col_type})')

        line = f'### Table {table_name}, columns = [{", ".join(columns)}]'
        output_lines.append(line)

    return "\n".join(output_lines)


def set_node_turn_n(node: NodeWithScore, turn_n: int):
    node.metadata["turn_n"] = turn_n
    return node


def parse_schemas_from_file(
        db_id: str,
        schema_path: Union[str, PathLike],
        output_format: str = "dataframe",
):
    schema_path = Path(schema_path) if isinstance(schema_path, str) else schema_path
    file_lis = get_sql_files(str(schema_path), ".json")

    all_schema = []
    for stem in file_lis:
        file_path = schema_path / db_id / f"{stem}.json"
        if not file_path.exists():
            continue
        col_info = load_dataset(file_path)
        assert isinstance(col_info, dict)
        meta_data = col_info["meta_data"]
        schema = {
            "Database name": meta_data["db_id"],
            "Table Name": meta_data["table_name"],
            "Field Name": col_info["column_name"],
            'Type': col_info["column_types"],
            'column_descriptions': col_info.get("column_descriptions", None),
            'sample_rows': col_info.get("sample_rows", None),
            'turn_n': 0
        }
        all_schema.append(schema)

    output_format = "dataframe" if not output_format else output_format
    if output_format == "dataframe":
        all_schema = pd.DataFrame(all_schema)

    return all_schema


def load_dataset(data_source: Union[str, PathLike]):
    data_source = Path(data_source) if isinstance(data_source, str) else data_source
    if not data_source.exists():
        warnings.warn("读取文件时，给定路径无效，该文件不存在！.", category=UserWarning)
        return None

    if data_source.suffix == ".json":
        with open(data_source, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    elif data_source.suffix in (".txt", ".sql"):
        with open(data_source, "r", encoding="utf-8") as f:
            dataset = f.read().strip()

    return dataset


def save_dataset(
        dataset: Union[str, List, Dict] = None,
        old_data_source: Union[str, PathLike] = None,  # todo: support both str or Path
        new_data_source: Union[str, PathLike] = None
):
    if old_data_source:
        dataset = load_dataset(old_data_source)
    assert dataset

    # 确保目录已创建
    assert new_data_source
    new_data_source = Path(new_data_source) if isinstance(new_data_source, str) else new_data_source

    new_data_source.parent.mkdir(parents=True, exist_ok=True)

    if new_data_source.suffix == ".json":
        with open(new_data_source, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
    elif new_data_source.suffix in ('.txt', '.sql'):
        with open(new_data_source, "w", encoding="utf-8") as f:
            f.write(dataset)
    elif new_data_source.suffix == ".csv":
        dataset.to_csv(str(new_data_source), index=False, encoding='utf-8')
    elif new_data_source.suffix == ".xlsx":
        dataset.to_excel(str(new_data_source), index=False)
