from os import PathLike

from llama_index.core.llms.llm import LLM
from pathlib import Path
from typing import Union


def summary_external_knowledge(
        question: str,
        llm: LLM,
        external_path: Union[str, PathLike] = None,
        external: str = None,  # 已读取的外部知识文档
        need_save: bool = True,
        save_path: Union[str, PathLike] = None,
):
    """ 从给定的外部知识文档中提取问题需要的外部知识 """
    if not external:
        assert external_path
        external_path = Path(external_path) if isinstance(external_path, str) else external_path
        assert external_path.exists()
        with open(external_path, "r", encoding="utf-8") as file:
            external = file.read()

    summary_external_knowledge_template = r"""
Generate a concise knowledge package strictly derived from the provided technical documentation to enable SQL generation for non-specialist users. Follow this precise workflow:

1. Semantic Anchoring
# Identify and index all domain-specific elements using quadruples:(Concept, Mathematical Representation, SQL Mapping, Documentation Section)
# Build dependency chains using → notation between logically connected concepts

2. Precision Filtering
# Apply three-stage relevance scoring:
a) Direct Match: Concepts with lexical overlap (+3 weight)
b) Contextual Neighbors: Co-occurring terms within 5 tokens (+2 weight)
c) Formula Components: Variables in target SQL's WHERE/JOIN clauses (+4 weight)

3. Compact Structuring
# Organize knowledge using SQL-aware hierarchy:
[Core Components]
• Essential definitions with explicit SQL mapping
• Key formulas → Convert to SQL-compatible constraints
[Derivation Toolkit]
• Subquery templates from documentation examples
• Unit-conversion factors (preserve original notations)
• Boundary conditions affecting JOIN/WHERE clauses

4. SQL-Ready Formatting
# Encode formulas as \boxed{{}} with SQL translation comments
# Mark subquery patterns with {{SQL}}
# Prefix each entry with [KD-#][Section X.Y]

### Validation Checklist:
a. Every SQL-relevant term has documentation-backed definition
b. Formula variables map to database schema elements
c. No standalone concepts - full dependency paths maintained
d. Maximum 12 knowledge units with dependency depth ≤3

### Example Output:
[KD-1][Section 3.2] Entropy Threshold
• System stability criterion: \boxed{{S < k\log W}}
→ {{SQL}} WHERE entropy < log(possible_states)*k
→ Links [KD-3][KD-7]

[KD-2][Section 4.1] Thermal Diffusion
• \boxed{{\alpha = \frac{{k}}{{\rho c_p}}}} (ρ: density [kg/m³])
→ Maps to JOIN material_properties ON temp_diffusivity = α

### Response Format:
[KD-#][Section X.Y] ConceptName
• Definition/Equation (Units)
→ {{SQL}} ImplementationNote
→ Linked [KD-#]

### Question
{question}

### External Knowledge Document
{external}

If no relevant content exists, output exactly: 'No Valuable External Knowledge'
# output:
"""
    prompt_ = summary_external_knowledge_template.format(question=question, external=external)
    try:
        summary = llm.complete(prompt_).text
        # Save external file if needed
        if need_save and save_path:
            save_path = Path(save_path) if isinstance(save_path, str) else save_path
            if not save_path.exists():
                raise Exception("The specified external knowledge document path does not exist!")
            with open(save_path, "w", encoding="utf-8") as file:
                file.write(summary)
        return summary
    except Exception as e:
        print(f"Error occurred in summary_external_knowledge: {e}")
        return None