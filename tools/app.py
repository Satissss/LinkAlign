from tools.SchemaLinkingTool import SchemaLinkingTool
from pipes.RagPipeline import *
from utils import *
import json
import pandas as pd
import os

base_dir = r".\spider2_dev\schemas"


# 将 nodes 解析为标准的 schema 输入，和 din-sql 对齐

def get_schema(
        db_id: str,
        question: str,
        similarity_top_k=15,
        open_agent: bool = False,
        instance_id: str = ""
):
    save_path = r".\spider2_dev\instance_schemas"
    # os.makedirs(save_path, exist_ok=True)

    file_name = instance_id
    file_name += "_agent" if open_agent else ""

    if os.path.isfile(rf"{save_path}\{file_name}.xlsx"):
        # return None
        df = pd.read_excel(rf"{save_path}\{file_name}.xlsx")
        return df

    vector_dir = rf"{base_dir}\{db_id}"

    vector_index = RagPipeLines.build_index_from_source(
        data_source=vector_dir,
        persist_dir=vector_dir + r"\vector_store",
        is_vector_store_exist=True,
        index_method="VectorStoreIndex"
    )
    retriever = RagPipeLines.get_retriever(index=vector_index, similarity_top_k=similarity_top_k)
    logger = Logger()
    if not open_agent:
        nodes_lis = SchemaLinkingTool.retrieve_complete(question=question,
                                                        retriever_lis=[retriever],
                                                        output_format="node",
                                                        open_locate=False
                                                        )
    else:
        nodes_lis = SchemaLinkingTool.retrieve_complete_by_multi_agent_debate(question=question,
                                                                              retriever_lis=[retriever],
                                                                              open_locate=False,
                                                                              output_format="node",
                                                                              logger=logger
                                                                              )
    logger.save(file_name=instance_id,
                save_path=r"D:\ScientificResearch\Text2Sql\spider2.0\spider2-lite\baselines\dinsql\preprocessed_data\spider2_dev\query_writing"
                )
    df = parse_schemas_from_nodes(nodes_lis)
    # 去重
    df = df.drop_duplicates()

    # save
    df.to_excel(rf"{save_path}\{file_name}.xlsx", index=False)

    # df = df.head(20)

    return df


if __name__ == "__main__":
    question = "Can you provide the number of distinct active and closed bike share stations for each year 2013 and 2014?"

    data = get_schema("austin", question, open_agent=True, instance_id="sf_bq058")
    print(data)
