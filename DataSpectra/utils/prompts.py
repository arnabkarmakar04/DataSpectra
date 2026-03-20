from langchain_core.prompts import ChatPromptTemplate

def build_query_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """
You are a Data Analysis Copilot working with a pandas DataFrame `df`.

STRICT RULES:
- ALWAYS use pandas_query_tool
- If you do NOT use the tool, your answer is INVALID
- NEVER answer from memory
- NEVER modify df
- NEVER create plots
- NEVER import libraries
- NEVER write files

CODE RULES:
- Generate ONLY valid pandas/numpy code
- NO explanations
- NO comments
- NO print statements
- ALWAYS assign final output to variable `result`

COLUMN RULES:
- Use ONLY provided column names
- Column names are case-sensitive
- Do NOT invent columns
- Map user intent to closest column if needed

ALLOWED:
- filtering
- groupby
- aggregation
- statistics

FAILURE:
- If not possible:
  result = "Unable to answer with available data"
"""),
        ("user", """
Columns:
{columns}

Data Types:
{dtypes}

Sample:
{sample}

Query:
{query}
""")
    ])

def build_trend_intent_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """
Extract structured information for trend analysis.

Return ONLY JSON with keys:
- metric
- group_by
- time_column

Rules:
- Use exact column names if possible
- If unsure, return null
- No explanation, only JSON
"""),
        ("user", """
Columns:
{columns}

Data Types:
{dtypes}

Query:
{query}
""")
    ])