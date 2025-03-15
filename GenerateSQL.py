# import debugpy; debugpy.connect(("127.0.0.1", 5688))
import pandas as pd
import time
import openai
from openai import OpenAI
import os
import sys
from tqdm import tqdm
import json
import logging
from multiprocessing import Pool, set_start_method
import concurrent.futures
from tools.SchemaLinkingTool import SchemaLinkingTool
from llms.qwen.QwenModel import QwenModel
from llms.deepseek.DeepseekModel import DeepseekModel
from utils import Logger

from prompts.din_prompt import *

# ------------------------------------------------------------------------------------------------------
QWEN_API_KEY = "sk-3f43effded4d49fc934829ee24e79f1d"
QWEN_MODEL = "deepseek-v3"  # "qwen-plus"  qwen2.5-coder-32b-instruct
open_agent = True
open_revise_from_feedback = True
similarity_top_k = 15

norm_llm = DeepseekModel(model_name="deepseek-chat", temperature=0.42)
# llm = YanModel(model_name="yantronic-o1", max_token=8192, temperature=0.42)
# llm = QwenModel(model_name="deepseek-v3", temperature=0.42, stream=True)
llm = DeepseekModel(model_name="deepseek-chat", temperature=0.42)

import argparse

parser = argparse.ArgumentParser(description="Process dataset and output file.")
parser.add_argument("--dev", type=str, default="spider2_dev")
parser.add_argument("--temperature", type=float, default=0.4)
parser.add_argument("--n", type=int, default=1)
parser.add_argument('--post_mode', type=str,
                    choices=['pass@n', 'consistency@n', 'consistency-from-generated-pass@n', None], default=None)
parser.add_argument('--processes', type=int, default=120)
parser.add_argument('--override', action='store_true')
parser.add_argument("--use_special_function", action="store_true", default=False)
parser.add_argument("--use_plan", action="store_true", default=False)

args = parser.parse_args()

DATASET_SCHEMA = f"./preprocessed_data/{args.dev}/tables_preprocessed.json"
DATASET = f"./preprocessed_data/{args.dev}/{args.dev}_preprocessed.json"

# openai.api_key = os.environ["OPENAI_API_KEY"]

special_function_info = "/* Potentially useful special functions with their usage: */\n" \
                        "{}"
plan_info = "/* A plan that is useful for guiding the generation of components of a complete SQL query: */\n" \
            "{}"


def check_length(existing_prompt, new, future_prompt):
    return len(existing_prompt + new + future_prompt) <= 1048576


def load_data(DATASET):
    return pd.read_json(DATASET)


def load_hard_prompt(instance_id):
    base_dir = r".\preprocessed_data\spider2_dev\reason_examples\one_shot"
    with open(rf"{base_dir}\{instance_id}.txt", "r", encoding="utf-8") as file:
        hard_examples = file.read()
    hard_examples = "\n[Examples Demonstration]\n" + hard_examples + "\n"
    return hard_examples


def hard_prompt_maker(row, schema_links, sub_questions, spider_schema, spider_foreign, args, dbms_name, external):
    test_sample_text, db_names = row['question'], row['db_id']

    # done: support multi-db
    if isinstance(db_names, str):
        db_names = [db_names]

    instruction = """[Instructions]
Use the intermediate representation, schema links, and the provided prior knowledge (including field and table information) to generate the correct SQL queries for each question. The SQL queries must be syntactically correct and logically aligned with the requirements of the question. 
You need to follow below requirements:
1. Understand the question: Carefully analyze the question to identify the relevant data and the required result.
2. Consult the schema: Use the schema links provided to identify the tables, fields, and relationships (including foreign keys and primary keys) necessary to answer the question.
3. Leverage prior knowledge: Utilize any domain-specific knowledge, field names, table relationships, and query logic to craft an accurate SQL query.
4. Use intermediate representations: Where applicable, break down the query into logical components such as CTEs (Common Table Expressions), subqueries, and joins, ensuring that each part of the query is clearly derived from the question and schema.
5. Adhere to DBMS syntax: Ensure that the SQL queries comply with the syntax specifications of {dbms_name}. Pay attention to common SQL conventions, such as SELECT, JOIN, WHERE, GROUP BY, and ORDER BY clauses, and ensure correct use of aggregate functions and data types.
6. Correct complex queries: For complex queries, use appropriate techniques (e.g., CTEs, subqueries) to avoid errors and improve readability.
7. Return only the SQL query: Provide the final, corrected SQL query without any explanations.
"""
    all_fields = ""
    for db_name in db_names:
        fields = find_fields_MYSQL_like(db_name, spider_schema)
        # foreign_keys = find_foreign_keys_MYSQL_like(db_name, spider_foreign)
        all_fields += fields + '\n'
        # all_foreign_keys += foreign_keys + '\n'
    stepping = f'\nA: Let\'s think step by step. Question can be solved by knowing the answer to the following sub-question "{sub_questions}".'

    existing_prompt = instruction + '\n[Question]: "' + test_sample_text + '"' + external
    existing_prompt += "\n[Provided Database Schema]: \n" + all_fields
    future_prompt = load_hard_prompt(row[
                                         'instance_id']) + '\n### Question: "' + test_sample_text + '\n### schema_links: ' + schema_links + stepping + '\n### Output SQL query"'

    if args.use_plan and row['plan'] is not None:
        new = plan_info.format(row['plan'])
        if check_length(existing_prompt, new, future_prompt):
            existing_prompt += new + '\n'
        else:
            print("Plan too long, skip. length: ", len(new))
    if args.use_special_function and row['special_function'] is not None:
        new = special_function_info.format(row['special_function'])
        if check_length(existing_prompt, new, future_prompt):
            existing_prompt += new + '\n'
        else:
            print("Special function too long, skip. length: ", len(new))

    existing_prompt += future_prompt
    return existing_prompt


def medium_prompt_maker(row, schema_links, spider_schema, spider_foreign, args, dbms_name, external):
    test_sample_text, db_names = row['question'], row['db_id']

    # done: support multi-db
    if isinstance(db_names, str):
        db_names = [db_names]

    instruction = """[Instructions]
Use the intermediate representation, schema links, and the provided prior knowledge (including field and table information) to generate the correct SQL queries for each question. The SQL queries must be syntactically correct and logically aligned with the requirements of the question. 
You need to follow below requirements:
1. Understand the question: Carefully analyze the question to identify the relevant data and the required result.
2. Consult the schema: Use the schema links provided to identify the tables, fields, and relationships (including foreign keys and primary keys) necessary to answer the question.
3. Leverage prior knowledge: Utilize any domain-specific knowledge, field names, table relationships, and query logic to craft an accurate SQL query.
4. Use intermediate representations: Where applicable, break down the query into logical components such as CTEs (Common Table Expressions), subqueries, and joins, ensuring that each part of the query is clearly derived from the question and schema.
5. Adhere to DBMS syntax: Ensure that the SQL queries comply with the syntax specifications of {dbms_name}. Pay attention to common SQL conventions, such as SELECT, JOIN, WHERE, GROUP BY, and ORDER BY clauses, and ensure correct use of aggregate functions and data types.
6. Correct complex queries: For complex queries, use appropriate techniques (e.g., CTEs, subqueries) to avoid errors and improve readability.
7. Return only the SQL query: Provide the final, corrected SQL query without any explanations.
"""

    all_fields = ""
    for db_name in db_names:
        fields = find_fields_MYSQL_like(db_name, spider_schema)
        foreign_keys = find_foreign_keys_MYSQL_like(db_name, spider_foreign)
        all_fields += fields + '\n'
        # all_foreign_keys += foreign_keys + '\n'

    # existing_prompt = instruction + '\n### Question: "' + test_sample_text + '"' + external
    # existing_prompt += "\n# Below is the whole of database schema: " + all_fields + all_foreign_keys
    # future_prompt = hard_prompt + '\nschema_links: ' + schema_links + stepping + '\n The SQL query for the sub-question"'

    existing_prompt = instruction + '\n[Question]: "' + test_sample_text + '"' + external
    existing_prompt += "\n#### [Provided Database Schema]: " + all_fields
    future_prompt = load_hard_prompt(row[
                                         "instance_id"]) + '\n#### [Question]: "' + test_sample_text + '"' '\n#### [Schema_links]: ' + schema_links + '\n# A: Let’s think step by step.'

    if args.use_plan and row['plan'] is not None:
        new = plan_info.format(row['plan'])
        if check_length(existing_prompt, new, future_prompt):
            existing_prompt += new + '\n'
        else:
            print("Plan too long, skip. length: ", len(new))
    if args.use_special_function and row['special_function'] is not None:
        new = special_function_info.format(row['special_function'])
        if check_length(existing_prompt, new, future_prompt):
            existing_prompt += new + '\n'
        else:
            print("Special function too long, skip. length: ", len(new))

    existing_prompt += future_prompt
    return existing_prompt


def easy_prompt_maker(test_sample_text, db_names, schema_links, spider_schema, spider_foreign):
    # done: support multi-db
    if isinstance(db_names, str):
        db_names = [db_names]

    instruction = "# Use the schema links to generate the SQL queries for each of the questions.\n"
    all_fields = ""

    for db_name in db_names:
        fields = find_fields_MYSQL_like(db_name, spider_schema)
        all_fields += fields + '\n'

    prompt = instruction + all_fields + easy_prompt + 'Q: "' + test_sample_text + '\nSchema_links: ' + schema_links + '\nSQL:'

    return prompt


def classification_prompt_maker(test_sample_text, db_names, schema_links, spider_schema, spider_foreign, external):
    # done: support multi-db
    if isinstance(db_names, str):
        db_names = [db_names]

    instruction = "# For the given question, classify it as NESTED. \n"
    instruction += "Break down the problem into subproblems and list them in the following format: questions = [q1,q2,q3..],e.g. questions = ['Which courses have prerequisite?']"
    instruction += "Always output Label: \"NESTED\".\n\n"

    all_fields = ""
    all_foreign_keys = "Foreign_keys = "
    for db_name in db_names:
        fields = find_fields_MYSQL_like(db_name, spider_schema)
        foreign_keys = find_foreign_keys_MYSQL_like(db_name, spider_foreign)

        all_fields += fields + '\n'
        all_foreign_keys += foreign_keys + '\n'
    if external is None:
        prompt = instruction + all_fields + all_foreign_keys + classification_prompt + '\nQuestion: "' + test_sample_text + '\nschema_links: ' + schema_links + '\nA: Let’s think step by step.'
    else:
        prompt = instruction + all_fields + all_foreign_keys + classification_prompt + '\nQuestion: "' + test_sample_text + "\n" + external + '\nschema_links: ' + schema_links + '\nA: Let’s think step by step.'

    return prompt


def schema_linking_prompt_maker(test_sample_text, db_names, spider_schema, spider_foreign):
    # done: support multi-db
    if isinstance(db_names, str):
        db_names = [db_names]

    # fields = find_fields_MYSQL_like(db_name)
    # foreign_keys = "Foreign_keys = " + find_foreign_keys_MYSQL_like(db_name, spider_foreign) + '\n'
    all_fields = ""
    all_foreign_keys = ""
    for db_name in db_names:
        fields = find_fields_MYSQL_like(db_name, spider_schema)
        foreign_keys = find_foreign_keys_MYSQL_like(db_name, spider_foreign)
        all_fields += fields + '\n'
        all_foreign_keys += foreign_keys + '\n'
    all_foreign_keys = "" if len(all_foreign_keys) == 0 else "Foreign_keys = " + all_foreign_keys

    instruction = "# Find the schema_links for generating SQL queries for each question based on the database schema and Foreign keys.\n"
    prompt = instruction + schema_linking_prompt + all_fields + all_foreign_keys + 'Q: "' + test_sample_text + """"\nA: Let’s think step by step."""
    return prompt


def find_fields_MYSQL_like(db_name, spider_schema):
    df = spider_schema[spider_schema['Database name'].str.lower() == db_name.lower()]
    df = df.groupby('Table Name')
    output = ""

    for name, group in df:
        output += f"Table: {name} Columns: ["
        fields = []
        for index, row in group.iterrows():
            field_details = f"{row['Field Name']} (Type:{row['Type']}) "
            fields.append(field_details)
        output += ", ".join(fields) + "]\n"

    return output.strip()


def find_primary_keys_MYSQL_like(db_name, spider_primary):
    df = spider_primary[spider_primary['Database name'] == db_name]
    output = "["
    for index, row in df.iterrows():
        output += row['Table Name'] + '.' + row['Primary Key'] + ','
    output = output[:-1]
    output += "]\n"
    return output


def find_foreign_keys_MYSQL_like(db_name, spider_foreign):
    df = spider_foreign[spider_foreign['Database name'] == db_name]
    output = "["
    for index, row in df.iterrows():
        output += row['First Table Name'] + '.' + row['First Table Foreign Key'] + " = " + row[
            'Second Table Name'] + '.' + row['Second Table Foreign Key'] + ','
    output = output[:-1] + "]"
    return output


def creating_schema(DATASET_JSON):
    # done: support multi-db
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(['column_names', 'table_names'], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index == -1:
                for table in tables:
                    schema.append([row['db_id'], table, '*', 'text'])
            else:
                schema.append([row['db_id'], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row['db_id'], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append([row['db_id'], tables[first_index], tables[second_index], first_column, second_column])
    spider_schema = pd.DataFrame(schema, columns=['Database name', ' Table Name', ' Field Name', ' Type'])
    spider_primary = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
    spider_foreign = pd.DataFrame(f_keys,
                                  columns=['Database name', 'First Table Name', 'Second Table Name',
                                           'First Table Foreign Key',
                                           'Second Table Foreign Key'])
    return spider_schema, spider_primary, spider_foreign


def debuger(test_sample_text, db_names, sql, spider_schema, spider_primary, spider_foreign, dbms_name, external):
    # done: support multi-db
    if isinstance(db_names, str):
        db_names = [db_names]

    instruction = f"""For the given question, use the provided tables, columns, foreign keys, and primary keys to fix the given SQL QUERY for any issues. If there are any problems, correct them. You should return the FIXED SQL QUERY only, without any explanation.
Use the following instructions for fixing the SQL QUERY:
1. Ensure that the database values mentioned in the question are explicitly used in the SQL.
2. Make sure the JOIN operations correctly use the Foreign Keys and ensure proper matching of the tables.
3. Use DESC and DISTINCT when needed to ensure proper ordering or uniqueness.
4. Verify the correct use of columns in the GROUP BY statement.
5. Pay attention to the columns in the SELECT statement to avoid missing or incorrect columns.
6. Only modify the GROUP BY clause when necessary (e.g., avoid redundant columns).
7. Use GROUP BY on only one column, unless multiple columns are explicitly required.
8. For complex SQL queries, ensure that CTEs (Common Table Expressions) are used where appropriate to avoid errors and simplify the query.
9. The SQL statements should strictly adhere to the syntax specifications of {dbms_name}. Ensure that the SQL can be executed without any syntax errors in a standard {dbms_name} environment."""
    all_fields = ""
    all_foreign_keys = "Foreign_keys = "
    all_primary_keys = "Primary_keys = "

    for db_name in db_names:
        fields = find_fields_MYSQL_like(db_name, spider_schema)
        foreign_keys = find_foreign_keys_MYSQL_like(db_name, spider_foreign)
        primary_keys = find_primary_keys_MYSQL_like(db_name, spider_primary)

        all_fields += fields + '\n'
        all_foreign_keys += foreign_keys + '\n'
        all_primary_keys += primary_keys + '\n'

    prompt = instruction + load_hard_prompt(row["instance_id"])
    prompt += all_fields + all_foreign_keys + all_primary_keys + '#### [Question] ' + test_sample_text + "\n"
    prompt += '#### [Existing SQL QUERY]\n' + sql + '\n#### Output FIXED SQL QUERY:'

    return prompt


def GPT4o_generation(prompt, n=1, model_name=None, temperature=0.45):
    try:
        client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=QWEN_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        response = client.chat.completions.create(
            model=QWEN_MODEL if model_name is None else model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            n=n,
            temperature=temperature,
            max_tokens=1000,  # 600
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["Q:"],
            timeout=75.0,
        )
        # return response['choices'][0]['message']['content']
        # (checked) copy from dailsql.
        response_clean = [choice.message.content for choice in response.choices]
        # if n == 1:
        #     response_clean = response_clean[0]

        # QWEN_MODEL = "qwen-max-0919"
        return dict(
            response=response_clean,
            **response.usage.dict()
        )


    except (Exception) as e:
        print(f"Error occurred: {e}")

        response = {"total_tokens": 0, "response": ["SELECT" for _ in range(n)]}
        raise e


def GPT4o_debug(prompt, n=1, model_name=None, temperature=0.45):
    try:
        client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=QWEN_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        response = client.chat.completions.create(
            model=QWEN_MODEL if model_name is None else model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            n=n,
            temperature=temperature,
            max_tokens=1000,  # 600
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["Q:"]
        )
        # return response['choices'][0]['message']['content']
        # (checked) copy from dailsql.
        response_clean = [choice.message.content for choice in response.choices]
        # if n == 1:
        #     response_clean = response_clean[0]
        return dict(
            response=response_clean,
            **response.usage.dict()
        )

    except (Exception) as e:
        print(f"Error occurred: {e}")
        response = {"total_tokens": 0, "response": ["SELECT" for _ in range(n)]}
        raise e


""" 当Spider Schema 规模庞大，难以将全部内容输入提示，因此需要进行检索和过滤，检索和过滤可以通过我们的框架实现。 """
""" 因此需要对现有 LinkAlign 方法进行调整，以适应单个大规模数据库的情况，第一步检索和第二部过滤 """


def dsr1_generation(prompt, n=1, model_name=None, temperature=0.45):
    try:
        client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=QWEN_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        response = client.chat.completions.create(
            model="deepseek-r1",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            n=n,
            temperature=temperature,
            max_tokens=8192,  # 600
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["Q:"],
            timeout=120.0
        )
        # return response['choices'][0]['message']['content']
        # (checked) copy from dailsql.
        response_clean = [choice.message.content for choice in response.choices]
        # if n == 1:
        #     response_clean = response_clean[0]

        # QWEN_MODEL = "qwen-max-0919"
        return dict(
            response=response_clean,
            **response.usage.dict()
        )


    except (Exception) as e:
        print(f"Error occurred: {e}")

        response = {"total_tokens": 0, "response": ["SELECT" for _ in range(n)]}
        raise e


def dsr1_debug(prompt, n=1, temperature=0.45):
    try:
        client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=QWEN_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        response = client.chat.completions.create(
            model="deepseek-r1",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            n=n,
            temperature=temperature,
            max_tokens=8192,  # 600
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["Q:"],
            timeout=120.0
        )
        # return response['choices'][0]['message']['content']
        # (checked) copy from dailsql.
        response_clean = [choice.message.content for choice in response.choices]
        # if n == 1:
        #     response_clean = response_clean[0]
        return dict(
            response=response_clean,
            **response.usage.dict()
        )

    except (Exception) as e:
        print(f"Error occurred: {e}")
        response = {"total_tokens": 0, "response": ["SELECT" for _ in range(n)]}
        raise e


def get_dbms_name(id):
    dbms_name = "BigQuery"
    if id.startswith("bq") or id.startswith("ga"):
        dbms_name = "BigQuery"
    elif id.startswith("local"):
        dbms_name = "sqlite"
    elif id.startswith("sf"):
        dbms_name = "SnowFlake"

    return dbms_name


def load_schema_linking(index, row, args, output_dir, spider_schema, spider_primary, spider_foreign, logger):
    save_path = "./preprocessed_data/spider2_dev/schema_links"
    instance_id = row["instance_id"]

    file_name = instance_id
    file_name += "_agent" if open_agent else ""

    if os.path.isfile(rf"{save_path}\{file_name}.txt"):
        with open(rf"{save_path}\{file_name}.txt", "r", encoding="utf-8") as f:
            schema_links = f.read()
        return schema_links.strip()

    schema_links = None
    while schema_links is None:

        try:
            all_fields = ""
            all_foreign_keys = ""
            fields = find_fields_MYSQL_like(row['db_id'], spider_schema)
            foreign_keys = find_foreign_keys_MYSQL_like(row['db_id'], spider_foreign)
            all_fields += fields + '\n'
            all_foreign_keys += foreign_keys + '\n'
            all_foreign_keys = "" if len(all_foreign_keys) == 0 else "Foreign_keys = " + all_foreign_keys
            context = all_fields + all_foreign_keys
            schema_links = SchemaLinkingTool.generate_by_multi_agent(llm=llm, query=row["question"],
                                                                     context=context,
                                                                     turn_n=1, linker_num=3, logger=logger
                                                                     )
            schema_links = schema_links.replace("`", "").replace("\n", "").replace("python", "")

        except Exception as e:
            print(e)
            raise e

    with open(rf"{save_path}\{file_name}.txt", "w", encoding="utf-8") as f:
        f.write(schema_links)

    return schema_links


def load_top_k(db_id):
    with open(r".\preprocessed_data\spider2_dev\db_info.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    lis = [row for row in data if row["db_id"] == db_id]
    count = lis[0]["count"]

    if count < 60:
        return count
    elif count < 90:
        return int(count / 4 * 3)
    else:
        return 70


def create_schema(db_id: str, question: str, instance_id: str):
    from tools.app import get_schema
    try:
        spider_schema = get_schema(db_id=db_id, question=question, similarity_top_k=similarity_top_k,
                                   open_agent=open_agent,
                                   instance_id=instance_id)
        # spider_schema = spider_schema.head(load_top_k(db_id))
    except:
        spider_schema = None
    spider_primary = pd.DataFrame(columns=['Database name', 'Table Name', 'Primary Key'])
    spider_foreign = pd.DataFrame(columns=['Database name', 'First Table Name', 'Second Table Name',
                                           'First Table Foreign Key',
                                           'Second Table Foreign Key'])
    return spider_schema, spider_primary, spider_foreign


def get_recent_files(directory, target_time):
    recent_files = []
    target_timestamp = target_time.timestamp()

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                mod_time = os.path.getmtime(file_path)
                if mod_time > target_timestamp:
                    recent_files.append(file)  # 只保存文件名
            except Exception as e:
                print(f"Error accessing {file_path}: {e}")

    return recent_files


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
            external = "\n####[External Priori Knowledge]:" + external + "\n"
            return external

    return None


def multi_processing_process_row(index, row, args, output_dir, spider_schema, spider_primary, spider_foreign, logger):
    # step1. schema linking
    logger.info("[DIN-SQL step1. schema linking]")

    schema_links = load_schema_linking(index, row, args, output_dir, spider_schema, spider_primary, spider_foreign,
                                       logger)

    external = load_external_knowledge(row["instance_id"])
    external = external if external else ""
    dbms_name = get_dbms_name(row["instance_id"])

    # # step2. difficulty classification
    logger.info("[DIN-SQL step2. difficulty classification]")
    try:
        class_prompt = classification_prompt_maker(row['question'], row['db_id'], schema_links[1:], spider_schema,
                                                   spider_foreign, external)
        classification = llm.complete(class_prompt).text
    except Exception as e:
        print(e)
        raise e
    logger.info("[classification]" + classification)

    try:
        predicted_class = classification.split("Label: ")[1]
    except:
        print("Slicing error for the classification module")
        predicted_class = '"NESTED"'
    logger.info("[predicted_class]" + predicted_class)

    # step3. SQL generation
    logger.info("[DIN-SQL step3. SQL generation]")
    try:
        sub_questions = classification.split('questions = [')[1].split(']')[0]
        flag = 'NESTED'
    except Exception as e:
        print('warning: error when parsing sub_question. treat it as Non-Nested. error:', e)
        flag = 'NON-NESTED'
    SQL_list = None

    while SQL_list is None:
        try:
            if flag == 'NESTED':
                hard_prompt_ = hard_prompt_maker(row, schema_links, sub_questions, spider_schema, spider_foreign,
                                                 args, dbms_name, external)
                sql = llm.complete(hard_prompt_).text
                SQL_list = [sql]
            else:
                medium_prompt_ = medium_prompt_maker(row, schema_links, spider_schema, spider_foreign, args,
                                                     dbms_name, external)
                sql = llm.complete(medium_prompt_).text
                SQL_list = [sql]
        except Exception as e:
            print(e)
            raise e

    logger.info("[SQL list]" + str(SQL_list[0]))

    # step4. SQL debugging and saving
    logger.info("[DIN-SQL step4. SQL debugging and saving]")
    for idx, sql in enumerate(SQL_list):
        try:
            if 'SQL' in sql:
                sql = sql.split("SQL:")[1]
            sql = sql.replace('```sql', '').replace('```', '').strip()
        except:
            print(f"SQL slicing error for index {idx}")
            sql = "SELECT"
        logger.info("[sql]" + sql)
        debugged_SQL = None
        while debugged_SQL is None:
            try:
                debug_prompt = debuger(row['question'], row['db_id'], sql, spider_schema, spider_primary,
                                       spider_foreign,
                                       dbms_name, external)
                debugged_SQL = llm.complete(debug_prompt).text

            except Exception as e:
                print(e)

        debugged_SQL = debugged_SQL.replace("\n", " ").replace('```sql', '').replace('```', '').strip()
        logger.info("[debugged_SQL]" + debugged_SQL)

        output_file = os.path.join(output_dir, f"{row['instance_id']}.sql")
        with open(output_file, 'w', encoding="utf-8") as f:
            f.write(debugged_SQL)

    return index


def process_task(input_args):
    index, row, args, output_dir = input_args
    spider_schema, spider_primary, spider_foreign = create_schema(row["db_id"], row["question"],
                                                                  row["instance_id"])
    input_args = (index, row, args, output_dir, spider_schema, spider_primary, spider_foreign)
    """ 线程任务函数 """
    if input_args[4] is None:
        return None

    instance_id = row["instance_id"]
    logger = Logger()

    try:
        multi_processing_process_row(*input_args, logger)
        return instance_id, None  # 成功处理返回 instance_id
    except Exception as e:
        return instance_id, str(e)  # 发生异常返回 instance_id 和错误信息


if __name__ == '__main__':
    set_start_method('spawn')
    s_time = time.time()
    val_df = load_data(DATASET)

    with open(DATASET, 'r', encoding="utf-8") as file:
        dev_data = json.load(file)

    output_dir = r'.\spider2_dev\predicted-SQL'

    # incremental prediction
    if args.override:
        pred_ids = set()
    else:
        pred_ids = set([file.split(".")[0].split("@")[0] for file in os.listdir(output_dir) if file.endswith(".sql")])

    error_lis = []

    inputs = []
    for index, row in val_df.iterrows():
        if row['instance_id'] not in pred_ids:
            inputs.append((index, row, args, output_dir))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:  # 根据实际情况调整 max_workers
        with tqdm(total=len(inputs)) as pbar:
            futures = {executor.submit(process_task, input_args): input_args for input_args in inputs}

    for future in concurrent.futures.as_completed(futures):
        try:
            instance_id, error = future.result()
            if error:
                error_lis.append({instance_id: error})
        except Exception as e:
            instance_id = futures[future]  # 获取失败任务的输入参数
            error_lis.append({instance_id: str(e)})  # 记录错误信息
            print(f"Error processing {instance_id}: {e}")  # 打印错误信息
        finally:
            pbar.update(1)  # 确保进度条更新更新进度条

    print(error_lis)
    print(f"Total time: {time.time() - s_time} seconds")
