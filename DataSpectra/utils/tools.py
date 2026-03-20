from langchain_core.tools import tool
import pandas as pd
import numpy as np
import json
# from utils.prompts import build_query_prompt
# from utils.prompts import build_trend_intent_prompt
from utils.prompts import *


def create_pandas_tool(df: pd.DataFrame, llm):
    prompt = build_query_prompt()

    @tool
    def pandas_query_tool(input: str) -> str:
        """Generate and execute pandas code to answer a query on dataframe df."""

        query = input

        chain = prompt | llm

        response = chain.invoke({
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_string(),
            "sample": df.head(2).to_markdown(),
            "query": query
        })

        code = response.content if hasattr(response, "content") else str(response)

        code = code.strip()

        if "```" in code:
            parts = code.split("```")
            if len(parts) > 1:
                code = parts[1]
            code = code.replace("python", "").strip()

        if not code.strip():
            return "Empty code generated"

        code_lower = code.lower()

        forbidden = [
            "import", "__", "open(", "exec(", "eval(",
            "to_csv", "to_excel", "plot", "plt",
            "matplotlib", "seaborn"
        ]

        forbidden = [x.lower() for x in forbidden]

        if any(x in code_lower for x in forbidden):
            return "Unsafe or disallowed operation detected"

        local_vars = {
            "df": df.copy(),
            "pd": pd,
            "np": np
        }

        try:
            exec(code, {}, local_vars)
            result = local_vars.get("result", "No result variable found")

            if isinstance(result, pd.DataFrame):
                return result.head(10).to_markdown()
            elif isinstance(result, pd.Series):
                return result.to_markdown()
            else:
                result_str = str(result)
                if len(result_str) > 2000:
                    return result_str[:2000] + "... (truncated)"
                return result_str

        except Exception as e:
            return f"Execution Error: {e}"

    return pandas_query_tool



def create_trend_tool(df: pd.DataFrame, llm):
    prompt = build_trend_intent_prompt()

    @tool
    def trend_analysis_tool(input: str) -> str:
        """Analyze trends using structured intent extraction."""

        query = input

        chain = prompt | llm

        response = chain.invoke({
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_string(),
            "query": query
        })

        content = response.content if hasattr(response, "content") else str(response)

        try:
            intent = json.loads(content)
        except:
            return "Failed to parse query for trend analysis"

        metric = intent.get("metric")
        group_col = intent.get("group_by")
        time_col = intent.get("time_column")

        df_local = df.copy()

        if metric not in df_local.columns:
            return f"Column '{metric}' not found"

        if group_col and group_col not in df_local.columns:
            group_col = None

        if time_col not in df_local.columns:
            for col in df_local.columns:
                try:
                    df_local[col] = pd.to_datetime(df_local[col])
                    time_col = col
                    break
                except:
                    continue

        if time_col is None:
            return "No valid time column found"

        df_local = df_local.sort_values(by=time_col)

        def compute_trend(series):
            if len(series) < 2:
                return "insufficient data"

            x = np.arange(len(series))
            slope = np.polyfit(x, series, 1)[0]

            if slope > 0:
                return f"increasing (slope={slope:.4f})"
            elif slope < 0:
                return f"decreasing (slope={slope:.4f})"
            else:
                return "stable"

        if group_col:
            results = []

            for group, sub_df in df_local.groupby(group_col):
                trend = compute_trend(sub_df[metric].dropna())
                results.append(f"{group}: {trend}")

            return "\n".join(results[:10])

        else:
            trend = compute_trend(df_local[metric].dropna())
            return f"Overall trend of {metric}: {trend}"

    return trend_analysis_tool