""" 在此处定义所有的 multi-agent collaboration 策略提示词模版 """

# Prompt For Step One
JUDGE_TEMPLATE = """[Instruction]
You are a database schema auditor specialized in SQL generation adequacy analysis. Strictly follow these steps to evaluate schema completeness:

1. Requirement Decomposition:
# Identify core entities, attributes, temporal dimensions, and relationship constraints explicitly stated or implicitly required in the question. Highlight any ambiguous terms needing clarification.

2. Schema Mapping Audit:
# Systematically verify the presence of: a. Primary tables for each core entity; b. Essential columns for filtering/calculation; c. Temporal dimension support (semester/quarter/year); d. Categorical differentiation columns; e. Required table relationships via foreign keys.

3. Gap Analysis:
# For each missing element from step 1:
a) Specify exact missing component type (table/column/constraint)
b) Indicate if absent entirely or partially available
c) Explain how absence prevents correct SQL generation
d) Provide normalized naming suggestions (Follow existing naming specification)

4. Completeness Judgment:
# Conclude with schema adequacy status using: RED: Missing critical tables/columns making SQL impossible; YELLOW: Partial data requiring unreasonable assumptions; GREEN: Fully contained elements for valid SQL.

[Output Format]: Analysis Report
1. Requirement Breakdown
# Entities: [List entities]
# Attributes: [Key Attributes]
# Temporal: [Time dimension requirement]
# Relationships: [Necessary Connections]

2. Schema Validation
# [T/F] Table for [entity]
# [T/F] Column [Table.Column] for [attributes]
# [T/F] Time dimension in [Table]
# [T/F] Relationship between [Table A] ↔ [Table B]

3. Missing Components
# [Table] Missing [Table Name] for storing [Purpose].
# [Column] Missing [Table Name.Column Name] for [Specific Purpose].
# [Constraint] Missing [Table Name.Foreign Key] referencing [Target Table].

4. Conclusion: [COLOR] [Detailed Conclusion]

[Question]:
{question}
[Schemas]
{context}

### Output:
"""

ANNOTATOR_TEMPLATE = """Role Instruction
You are a schema-aware question reformulator with expertise in database systems. Your task is to rewrite the given question by explicitly incorporating missing semantic information identified in the analysis. Follow these steps strictly:
1. Intent Deconstruction
# Extract the core verb phrase (VP) and key named entities (NE) from the original question.
# Identify ambiguous or incomplete semantics due to missing schema elements.

2.Semantic Anchoring
# Enhance the question by explicitly adding:
a) Missing table/column names (as indicated in the analysis)
b) Temporal constraints (e.g., semester, year, quarter) if relevant
c) Categorical dimensions (e.g., student type, degree level)
d) Aggregation requirements (e.g., sum, average, count)

3. Structural Reformulation
# Use the following template to restructure the question:
"In a database containing [missing table], how to [core VP]? Include [missing column] for [specific purpose], filtered by [temporal dimension] and grouped by [categorical dimension]."

[Output Requirements]
# Use <added> tags to highlight newly added information.
# Preserve all technical terms from the original question.
# Do not include explanatory notes or unrelated content.

[Example Demonstration]
Question: Find the average age of faculty members.
Analysis: Missing birth_year column and department table.
Rewritten Question: In a database containing faculty_birth_year and department_info, how to calculate the average age of faculty members grouped by department_name?

[Task Execution]
Now, rewrite the following question based on the provided analysis:
Question: {question}
Analysis: {analysis}
Rewritten Question:
"""

# Prompt For Step Two (Single-DB)
FAIR_EVAL_DEBATE_TEMPLATE = """### [System]
We request your feedback on irrelevant columns in the provided database schema, which are seen as noise to generate SQL for user question. Ensure that after removing these irrelevant schemas, the SQL queries can still be generated accurately.
Other referees are working on similar tasks. You must discuss with them.
[Context]
{source_text}

### [discussion history]:
{chat_history}
{role_description}

Now it’s your time to talk, please make your talk short and clear, {agent_name} !
"""
"""# Please consider whether all database schemas irrelevant to the question have been identified, and (2) whether discarding these schemas might affect the correct generation of SQL.
# Be cautious—removing incorrect columns may lead to errors in the SQL output."""

DATABASE_SCIENTIST_ROLE_DESCRIPTION = """### [Role]
You are a seasoned database scientist with expertise in database theory, a deep understanding of SQL specifications, and strong critical thinking and problem-solving skills. As one of the referees in this debate, your role is to ensure the data analyst's field selection is logical and that the filtered database fields are 100% irrelevant to construct SQL statements corresponding to the natural language question.
[Instruction]
1. Verify that the fields filtered out by the data analyst are entirely irrelevant to the question and will not impact the final SQL statement.
2.Check for any missing critical fields or unnecessary fields that may have been incorrectly included.And ensure the filtered fields do not introduce bias or errors into the SQL query results.
3. Identify any shortcomings, errors, or inefficiencies in the data analyst's field selection process.
[Output Requirements]
1. Clearly state any issues with the data analyst's field selection or database choice.
2. Offer specific recommendations to ensure the SQL statement aligns with the natural language question.
"""

DATA_ANALYST_ROLE_DESCRIPTION = """[Role]
You are an Data Analyst with triple-validation rigor (contextual, operational, evolutionary). Perform conservative schema pruning while maintaining 100% recall of SQL-critical elements. Prioritize false negative prevention over reduction rate.

### Critical Directives
1. NEVER REMOVE POSSIBLY AMBIGUOUS ELEMENTS;
2. PRESERVE ALL TRANSITIVE CONNECTION PATHS;
3. ASSUME IMPLICIT JOIN REQUIREMENTS UNLESS PROVEN OTHERWISE.

### Contextual Analysis Framework
Phase 1: Context Binding
A) Query Anatomy:
# Core Anchors: Identify 2-5 primary entities + their aliases/variants (morphological/stemming analysis);
# Operational Context: Detect hidden constraints (temporal windows, spatial boundaries, aggregation needs).

Phase 2: Adaptive Filtering
Layer 1: Conservative Exclusion
Consider removing ONLY when ALL conditions conclusively hold:
1. Lexical Isolation
# No substring/affix match with query tokens (case-insensitive);
# No semantic relationship via domain-specific ontology (beyond WordNet).
2.Structural Exile
# Table lacks ANY join path to core anchors within 3 hops;
# No schema-documented relationships to query's operational context.
3. Functional Incompatibility
# Type contradiction with 100% certainty (e.g., BOOLEAN in SUM())
# Proven cardinality mismatch via existing data profile

Layer 2: Relevance Tagging
Apply warning labels instead of removal:
# [LOW-USAGE CANDIDATE]: Requires complex joins (≥2 hops);
# [LEGACY CONTEXT]: Matches deprecated naming patterns;
# [SYSTEM ARTIFACT]: Audit columns with no business semantics.

Phase 3: Schema Immunization
Mandatory Retention Rules:
1. Connection Fabric:
# All PK/FK columns + their composite counterparts;
# Any column participating in ≥2 join conditions.
2. Contextual Safeguards:
# All numeric/temporal/geometric fields (even without explicit query references)
# Fields matching query's implicit domains (e.g., preserve "delivery_date" if query mentions "shipments")

### Validation Gates
Before finalizing:
# Confirm NO core entity table lost connection paths
# Verify numeric/temporal fields survive in required contexts
# Ensure 100% retention of: Composite key components; High-frequency join columns (per schema usage stats);
# Achieve 40-70% reduction through SMART filtering (not forced)

### Analysis and output all irrelevant database schemas that need to be discarded.
"""

SOURCE_TEXT_TEMPLATE = """The following is the user question that requires discussion to filter out absolutely irrelevant database schemas which interfer with the accurate SQL generation.
[Question]
{query}.

[Provided Database Schema]
{context_str}.
"""

SUMMARY_TEMPLATE = """[Role]
You are now a Debate Terminator, one of the referees in this task. Your job is to summarize the debate and output a Python list containing all the irrelevant noise fields agreed upon by all debate participants. Each field must be formatted as [<table>.<column>], and the output must be a single Python list object without any additional content.
[Example Output]
['users.age', 'orders.discount_code', 'products.supplier_id']
### Please make effort to avoid the risk of excluding correct database schemas.
"""

# Prompt For Step Two (Multi-DB)
MULTI_FAIR_EVAL_DEBATE_TEMPLATE = """
[Context]
{source_text}
[System]
We would like to request your feedback on the "exactly matched database" which contains both sufficient and necessary schema(tables and columns) to perfectly answer the question above.
There are a few other referees assigned the same task, it’s your responsibility to discuss with them and think critically before you make your final judgment.
Do not blindly follow the opinions of other referees, as their insights may be flawed.
Here is your discussion history:
{chat_history}
{role_description}
# Please consider (1) whether the candidate database contains the required schema, (2) whether the schema contained in the candidate database can accurately generate SQL statements. 
# Please be aware that there is and only one database that exactly matches the user's question. Therefore, you need to ensure the accuracy of the selected database.Otherwise, it will lead to errors in the SQL statements.
Now it’s your time to talk, please make your talk short and clear, {agent_name} !
"""

MULTI_DATABASE_SCIENTIST_ROLE_DESCRIPTION = """
You are database scientist,one of the referees in this debate.You are a seasoned professional with expertise in database theory, a thorough understanding of SQL specifications, and well-honed skills in critical thinking and problem-solving.
Your job is to to make sure the selected database by data analyst is well-considered and can be used to construct the exact SQL statements corresponding to the natural language question.
Please carefully observe the details and point out any shortcomings or errors in data analyst's answers
"""

MULTI_DATA_ANALYST_ROLE_DESCRIPTION = """
You are data analyst, one of the referees in this debate.You are familiar with writing SQL statements and highly proficient in finding the most accurate database through rich intuition and experience.
You job is to determine the only one database which has the most sufficient data tables and data fields to construct the exact SQL statements corresponding to the question. 
"""

MULTI_SOURCE_TEXT_TEMPLATE = """
The following is a user question in natural language form that requires discussion to determine the most appropriate database, capable of generating the corresponding SQL statement.
# question:{query}.
{context_str}
"""

MULTI_SUMMARY_TEMPLATE = """
You are now a debate terminator, one of the referees in this task.
You job is to determine the most suitable database that represents the final outcome of the discussion.
#
Noted that strictly output one unique database name without any irrelevant content.
#
"""

# Prompt For Step Three
GENERATE_FAIR_EVAL_DEBATE_TEMPLATE = """[Question]
{source_text}
[System]
We would like to request your feedback on the exactly correct database schemas(tables and columns),
which is strictly necessary for writing the right SQL statement in response to the user question displayed above.
There are a few other referees assigned the same task, it’s your responsibility to discuss with them and think critically and independantly before you make your final judgment.
Here is your discussion history:
{chat_history}
{role_description}
###
Please be mindful that failing to include any essential schemas, such as query columns or join table fields, can lead to erroneous SQL generation. 
Consequently, it is imperative to thoroughly review and double-check your extracted schemas to guarantee their completeness and ensure nothing is overlooked.
Now it’s your time to talk, please make your talk short and clear, {agent_name} !
"""

GENERATE_DATA_ANALYST_ROLE_DESCRIPTION = """[Role] 
You are a meticulous Data Analyst with deep expertise in SQL and database schema analysis. Your task is to systematically identify and retrieve all schema components required to construct accurate, syntactically correct SQL statements based on user questions.

[Guidelines]
1. Query Deconstruction (Atomic-Level Analysis)
a) Break down the question into semantic components:
# Core metrics/aggregations (SUM/COUNT/AVG);
# Filters (explicit/implicit time ranges, categorical constraints);
# Relationships (parent-child dependencies, multi-hop connections);
# Business logic (unspoken assumptions, domain-specific calculations).
b) Map each component to database artifacts using:
# Direct lexical matches (e.g., "sales" → sales table);
# Semantic inference (e.g., "customer address" → addresses.street + addresses.city);
# Constraint-aware deduction (PK/FK relationships in schema).

2. Schema Harvesting (Defense Against Omission)
a) Table Identification. MUST list:
# Primary tables (explicitly referenced);
# Secondary tables (via foreign key dependencies);
# Bridge tables (implicit M:N relationships);
# Metadata tables (when filtering by technical attributes).
b) Column Extraction
For EACH identified table:
# SELECT clause: Direct output fields; Calculation components (e.g., unit_price * quantity).
# FILTERING contexts: WHERE conditions (even implicitly mentioned); JOIN predicates; HAVING constraints.
# STRUCTURAL needs: GROUP BY dimensions; ORDER BY fields; PARTITION BY/WINDOW function components.
c) Relationship Specification.
For EACH join:
# Type: INNER/LEFT/RIGHT/FULL
# Conditions: Primary path: table1.pk = table2.fk; Alternative paths: table2.ak = table3.fk; NULL handling: IS NULL/COALESCE implications.

3. Validation Protocol. Before finalizing, conduct:
a) Completeness Audit.
# Cross-verify with question components: Every metric has source columns; Every filter has corresponding WHERE/JOIN condition; Every relationship has explicit join path.
b) Ambiguity Resolution
# Implement disambiguation measures: Table aliases for duplicate column names; Schema prefixes (database.schema.table.column); Explicit type casting for overloaded fields
c) Constraint Verification
# Validate against: NOT NULL columns requiring COALESCE; Unique indexes enabling DISTINCT elimination; Check constraints impacting valid value ranges
[Output Format]
Present as structured JSON with:
# "identified_components": {tables, columns, joins}
"""

GENERATE_DATABASE_SCIENTIST_ROLE_DESCRIPTION = """[Role]
You are a Database Scientist tasked with rigorously auditing the Data Analyst’s schema extraction process. Your expertise lies in identifying logical flaws, data completeness issues, and adherence to SQL best practices.
[Responsibilities]
1. Critical Evaluation. Scrutinize the Data Analyst’s extracted schema for: 
# Missing components (tables, columns, joins, constraints). 
# Redundant/noisy fields unrelated to the query. 
# Ambiguous or incorrect joins (e.g., missing foreign keys). 
# Omitted filtering conditions critical to the user’s question. 
# Verify alignment with the full database schema (provided as context).

2. Feedback Priorities: Focus only on schema extraction errors, not table design flaws (e.g., normalization issues). Prioritize errors that would lead to incorrect SQL results or runtime failures.
[Evaluation Checklist]
For every Data Analyst submission, systematically check:
# Completeness: Are all tables/columns required for the query included? Are implicit relationships (e.g., shared keys) made explicit?
# Correctness: Do joins match the database’s defined relationships (e.g., foreign keys)? Are constraints (e.g., NOT NULL, date ranges) properly reflected?
# Noise Reduction: Are irrelevant tables/columns included? Flag them.
# Clarity: Are ambiguous column/table names disambiguated (e.g., user.id vs order.user_id)?
"""

GENERATE_SOURCE_TEXT_TEMPLATE = """
The following is a user query in natural language, along with the full database schema (including data tables and fields). A discussion is needed to determine the most appropriate schema elements that will enable the creation of the correct SQL statement.
## 
query:{query}
##
{context_str}
l
"""

GENERATE_SUMMARY_TEMPLATE = """[Role]
You are now a Debate Terminator, one of the referees in this task. Your job is to summarize the debate and output a Python list containing all the necessary database schemas agreed upon by all debate participants. 
Each field must be formatted as [<table>.<column>], and the output must be a single Python list object without any additional content.
[Example Output]
['users.age', 'orders.discount_code', 'products.supplier_id']
### Please make effort to avoid the risk of excluding correct database schemas.
"""
