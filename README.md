## LinkAlign: Scalable Schema Linking for Real-World Large-Scale Multi-Database Text-to-SQL

## Introduction

Schema linking is a critical bottleneck in achieving human-level performance in Text to-SQL tasks, particularly in real-world large scale multi-database scenarios. Addressing schema linking faces two major challenges: (1) Database Retrieval: selecting the correct database from a large schema pool in multi database settings, while filtering out irrele vant ones. (2) Schema Item Grounding: ac curately identifying the relevant tables and columns from within a large and redundant schema for SQL generation. To address this, we introduce LinkAlign, a novel framework that can effectively adapt existing baselines to real-world environments by systematically ad dressing schema linking. Our framework com prises three key steps: multi-round semantic enhanced retrieval and irrelevant information isolation for Challenge 1, and schema extrac tion enhancement for Challenge 2. We evalu ate our method performance of schema linking on the SPIDER and BIRD benchmarks, and the ability to adapt existing Text-to-SQL mod els to real-world environments on the SPIDER 2.0-lite benchmark. Experiments show that LinkAlign outperforms existing baselines in multi-database settings, demonstrating its effec tiveness and robustness. On the other hand, our method ranks highest among models excluding those using long chain-of-thought reasoning LLMs. This work bridges the gap between current research and real-world scenarios, pro viding a practical solution for robust and scal able schema linking. 

![Overview1](assets/Overview1.png)

## Requirements

* sentence-transformers==3.0.1
* transformers==4.42.4
* torch==2.4.0
* llama-index==0.10.62
* llama-index-core==0.10.62
* llama-index-embeddings-huggingface==0.1.5
* openai==1.41.0

## Local Deployment

We slightly modified the LlamaIndex Framework to better develop our project.

```python
# 首先进入当前虚拟环境管理的 LlamaIndex 目录
cd .../site-packages/llama_index

vim embeddings/huggingface/base.py
# 注释第 87 行 safe_serialization 参数
model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_folder,
                trust_remote_code=trust_remote_code,
                # safe_serialization=safe_serialization,
            )

vim core/indices/vector_store/retrievers/retriever.py
# 在VectorIndexRetriever 类的 __init__ 方法中增加 self._orininal_ids 属性
self._orininal_ids = node_ids

# 增加下面三个成员方法
@property
def index(self) -> VectorStoreIndex:
    """ return object of VectorStoreIndex """
    return self._index

def change_node_ids(self, node_ids):
    ids_ = []
    if self._node_ids:
       ids_ = self._node_ids
    else:
        doc_info_dict = self._index.ref_doc_info
        for key, ref_doc_info in doc_info_dict.items():
            ids_.extend(ref_doc_info.node_ids)

        self._node_ids = [id_ for id_ in ids_ if id_ in node_ids]

def back_to_original_ids(self):
    ids_ = []
    if self._orininal_ids:
        ids_ = self._orininal_ids
    else:
        doc_info_dict = self._index.ref_doc_info
        for key, ref_doc_info in doc_info_dict.items():
            ids_.extend(ref_doc_info.node_ids)
                
    self._node_ids = ids_
```

