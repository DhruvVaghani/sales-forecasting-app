import joblib
from xgboost import XGBRegressor
# Load encoder
le_family = joblib.load("le_family.pkl")
# Load the model
model = joblib.load("xgboost_model.pkl")

# Now you can predict
# Example: model.predict(X) where X is a DataFrame of features
from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict,Optional, Any
import os
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain.tools import Tool

openai_key = st.secrets["API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_key



db = SQLDatabase.from_uri("sqlite:///mydatabase (1).db")
print(db.dialect)
print(db.get_usable_table_names())


### THIS IS TO INITIALIZE THE STATE
class State(TypedDict, total= False):
    question: str
    sql_query: str
    sql_result: Any
    final_answer: str

##INITIALIZE CHAT BOT MODEL 
llm = init_chat_model("gpt-4o", model_provider="openai")

from langchain_core.prompts import ChatPromptTemplate

system_message = """
You are a helpful assistant that writes SQL queries for a SQLite database.

Only use the table named `forecasted_sales`.
NEVER reference or use any other table like `sales_forecast`, `family`, or `families`.

Important: The column `family` is stored as a label-encoded INTEGER, not a string.
That means you should NEVER write queries like `family = 'EGGS'`.
Just write the SQL assuming the family is already encoded (e.g., `family = 4`) 
and the system will replace that value for you using preprocessing.

Only use columns from forecasted_sales: date, store_nbr, family, predicted_sales.

Always use LIMIT {top_k} unless the user specifies a number of rows.
Always filter with WHERE clauses on date, store_nbr, or family as appropriate.
Use single quotes only for string fields like `date` (e.g., WHERE date = '2017-08-16').
"""


user_prompt = "Question: {input}"

# query_prompt_template = ChatPromptTemplate(
#     [("system", system_message), ("user", user_prompt)]
#     )

from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate

# Sample few-shot examples (feel free to adjust family numbers to match your encoder)
examples = [
    {
        "input": "predict sales for store 1, family EGGS on 2017-08-16",
        "query": "SELECT predicted_sales FROM forecasted_sales WHERE store_nbr = 1 AND family = 4 AND date LIKE '2017-08-16%'"
    },
    {
        "input": "show forecasted sales for store 2, family MILK on 2017-09-01",
        "query": "SELECT predicted_sales FROM forecasted_sales WHERE store_nbr = 2 AND family = 7 AND date LIKE '2017-09-01%'"
    },
    {
    "input": "Show predicted sales for store 1, family EGGS between 16 December 2017 and 22 December 2017.",
    "query": "SELECT predicted_sales FROM forecasted_sales WHERE store_nbr = 1 AND family = 10 AND date BETWEEN '2017-08-16' AND '2017-08-22'"
    },
    {
    "input": "Show sales trend for store 1, family BREAD over September 2017",
    "query": "SELECT date, predicted_sales FROM forecasted_sales WHERE store_nbr = 1 AND family = 3 AND date BETWEEN '2017-08-01' AND '2017-09-30'"
}

]

# Convert examples to prompt format
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{query}")
])

few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

# Final combined prompt: instructions + few-shot + user input
query_prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_message),
    few_shot,
    ("human", "Question: {input}")
])



def preprocess_sql_query(query: str) -> str:
    """
    Rewrites SQL queries to replace family names with encoded integers,
    and fixes TIMESTAMP date format by converting `date = 'YYYY-MM-DD'` to `date LIKE 'YYYY-MM-DD%'`.
    """
    import re

    # Convert family = 'XYZ' to family = <int>
    for fam in le_family.classes_:
        pattern = re.compile(rf"(family\s*=\s*)['\"]{fam}['\"]", re.IGNORECASE)
        if pattern.search(query):
            encoded_value = int(le_family.transform([fam])[0])
            query = pattern.sub(lambda m: f"{m.group(1)}{encoded_value}", query)

    # Convert date = 'YYYY-MM-DD' to date LIKE 'YYYY-MM-DD%'
    query = re.sub(r"(date\s*=\s*)'(\d{4}-\d{2}-\d{2})'", r"date LIKE '\2%'", query)


    # Clean any trailing punctuation or stray characters
    query = query.strip().strip(";").strip('"').strip("'")

    return query







from typing_extensions import Annotated


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

## # This function is a LangGraph node that takes a user's natural language question,
# generates a SQL query using a structured LLM prompt, and returns that query.
# It uses the current database dialect and schema to guide SQL generation.
# The output is structured using a TypedDict to ensure clean, valid queries,
# which will be executed in the next step of the workflow.

def write_query(state: State):
    """Generate SQL query to fetch information. Do not create querries with multiple join and nested queries"""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"sql_query": result["query"]}


# This function is a LangGraph node responsible for executing the SQL query 
# generated in the previous step. It uses LangChain's QuerySQLDatabaseTool to 
# safely run the query on the connected SQLite database. The result is stored 
# in the state for downstream processing (like converting to a natural language answer).
# Note: Only SELECT queries should be allowed to ensure read-only access.

def execute_query(state: State):
    """Execute SQL query with label-encoded preprocessing."""
    cleaned_query = preprocess_sql_query(state["sql_query"])
    print("🧪 Final SQL query:", cleaned_query)  # optional debug
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"sql_result": execute_query_tool.invoke(cleaned_query)}

##Generate answer
##Finally, our last step generates an answer to the question given the information pulled from the database:


def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["sql_query"]}\n'
        f'SQL Result: {state["sql_result"]}'
    )
    response = llm.invoke(prompt)
    return {"final_answer": response.content}

##Orchestrating with LangGraph
###Finally, we compile our application into a single graph object. In this case, we are just connecting the three steps into a single sequence.

from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()



from typing_extensions import Annotated
from typing import TypedDict

class ForecastRequest(TypedDict):
    store_nbr: Annotated[int, "Store number as an integer"]
    family: Annotated[str, "Product family name"]
    date: Annotated[str, "Forecast date in YYYY-MM-DD format"]



# structured_llm = llm.with_structured_output(ForecastRequest)

# extracted_params = structured_llm.invoke(
#     "Can you give me forecast for store 5, family 3, sometime in Jan 2026?"
# )


### FALLBACKK TOOL 


### FALLBACKK FUNCTION  

import sqlite3
import pandas as pd
from datetime import datetime

def predict_and_store(question: str) -> str:
    structured_llm = llm.with_structured_output(ForecastRequest)
    try:
        params = structured_llm.invoke(question)
    except Exception:
        return "Sorry, I couldn't extract the required forecast parameters from your query."

    store_nbr = params["store_nbr"]
    family = params["family"].upper().strip()
    date = params["date"]

      # Encode family string to match model expectation#
    try:
        encoded_family = le_family.transform([family])[0]
    except ValueError:
        return f"Sorry, '{family}' is not a valid product family. Try one of: {list(le_family.classes_)}"


    # --- 2. Feature Engineering (simplified placeholder)
    features = pd.DataFrame([{
        'store_nbr': store_nbr,
        'family': encoded_family,
        'onpromotion': 0,
        'transactions': 200,             # Simulated
        'oil_price': 55.2,               # Simulated
        'lag_1': 13500,                  # Simulated
        'lag_7': 13200,                  # Simulated
        'rolling_mean_7': 13400,         # Simulated
        'day': int(date[-2:]),
        'month': int(date[5:7]),
        'year': int(date[:4]),
        'dayofweek': datetime.strptime(date, "%Y-%m-%d").weekday(),
        'city': 1,
        'state': 2,
        'type': 0,
        'cluster': 3
    }])

    # --- 3. Predict using your XGBoost model
    predicted_sales = model.predict(features)[0]

    # --- 4. Insert into the forecasted_sales DB table
    conn = sqlite3.connect("mydatabase (1).db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO forecasted_sales (date, store_nbr, family, predicted_sales)
        VALUES (?, ?, ?, ?)
        """,
        (date, store_nbr, encoded_family, round(predicted_sales, 2))
    )
    conn.commit()
    conn.close()

    # --- 5. Return result
    return f"The predicted sales for store {store_nbr}, family {family} on {date} is {round(predicted_sales, 2)} units."




## WRAPPING THE FUNCTION CREATE ABOVE TO MAKE A TOOL ##


forecast_tool = Tool(
    name="LiveForecast",
    func=predict_and_store,
    description=(
        "Use this tool when no forecast exists in the mydatabase (1) table forecasted_sales. "
        "It takes a user question about future sales, generates features, "
        "Generate only one simple querry no subquerries "
        "uses a machine learning model to predict sales, and stores the result in the mydatabase table name forecasted_sales. Columns:"
        "- date (TIMESTAMP)"
        "- store_nbr (INTEGER)"
        "- family (INTEGER)"
        "- predicted_sales (REAL)"
    )
)




from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
import logging
logging.basicConfig(level=logging.INFO)

# def safe_query_tool_func(sql: str):
#     cleaned = preprocess_sql_query(sql)
#     logging.info(f"🧪 Cleaned SQL: {cleaned}")
#     return db.run(cleaned)

# from langchain.tools import Tool



def safe_query_tool_func(sql: str):
    cleaned = preprocess_sql_query(sql)
    print("🧪 Cleaned SQL:", cleaned)
    df = pd.read_sql(cleaned, sqlite3.connect("mydatabase (1).db"))
    
    # Optional: format output
    preview = df.to_dict(orient="records")
    return {
        "output": f"Here are the results for your query:",
        "sql_query": cleaned,
        "fallback_used": False,
        "df": df
    }

safe_query_tool = Tool(
    name="sql_db_query",
    func=safe_query_tool_func,
    
    description="Safely query the forecasted_sales table by preprocessing label-encoded fields like family."
)

from langchain.agents import initialize_agent

agent = initialize_agent(
    tools=[ forecast_tool, safe_query_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True
)

if __name__ == "__main__":
    print(agent.invoke("predict sales for store number 1, family EGGS on 17th December 2017"))


# def run_agent(query: str) -> dict:
#     result = agent.invoke(query)
#     # This assumes your tools are returning dictionaries
#     return {
#         "output": result.get("output", result),
#         "sql_query": result.get("sql_query", "N/A"),
#         "fallback_used": result.get("fallback_used", False)
#     }


def run_agent(query: str) -> dict:
    try:
        result = agent.invoke(query)
        if isinstance(result, str):
            return {"output": result, "sql_query": "N/A", "fallback_used": False, "df": None}
        return result
    except Exception as e:
        # If query fallback tool returns raw data
        if hasattr(e, "args") and isinstance(e.args[0], dict):
            return e.args[0]
        return {"output": str(e), "sql_query": "N/A", "fallback_used": False, "df": None}
