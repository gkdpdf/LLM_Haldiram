from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
db = SQLDatabase.from_uri("postgresql+psycopg2://postgres:12345678@localhost:5432/haldiram")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

with open("annotated_schema.md", "r", encoding="utf-8") as f:
    annotated_schema = f.read()

with open("relationship.txt", "r", encoding="utf-8") as f:
    relationships = f.read()

sql_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a world-class SQL expert. 
Convert natural language into **valid PostgreSQL queries** only.

### Available Schema:
{annotated_schema}

### Database Relationships:
{relationships}

CRITICAL RULES:
- Use ONLY listed tables/columns in the schema
- Apply relationships correctly when joining
- Use EXACT equality (=) for filters, NOT LIKE or ILIKE
- Include aggregations (SUM, COUNT, etc.) for sales/metric queries
- Do NOT assume or create any columns/tables
- Return the final SQL in a ```sql block
"""),
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

sql_agent = create_openai_functions_agent(
    llm=llm,
    tools=toolkit.get_tools(),
    prompt=sql_prompt
)

sql_agent_executor = AgentExecutor(agent=sql_agent, tools=toolkit.get_tools(), verbose=True)


class GraphState(TypedDict, total=False):
    user_query: str
    catalog: Dict[str, Any]              
    table_columns: Dict[str, List[str]]  
    annotated_schema: str
    relationships: str
    resolved: Dict[str, Any]
    sql_result: Any                      
    validated_sql: str                  
    validation_status: str               
    validation_error: Optional[str]
    execution_result: Any                
    execution_status: str               
    execution_error: Optional[str]
    route_decision: str                
    final_output: str                    
    reasoning_trace: List[str]


def sql_agent_node(state: GraphState):
    resolved = state["resolved"]
    annotated_schema = state["annotated_schema"]
    relationships = state['relationships']
    user_query = state['user_query']
    
    # Extract exact resolved entity values for precise filtering
    entity_filters = []
    
    # Product filter
    if resolved.get('entities', {}).get('product'):
        product_data = resolved['entities']['product']
        if isinstance(product_data, dict):
            product_value = product_data.get('value')
            product_column = product_data.get('column', 'product_name')
            product_table = product_data.get('table')
            if product_value:
                entity_filters.append(
                    f"MUST filter: {product_column} = '{product_value}' (exact match, NOT LIKE)"
                )
    
    # Distributor filter
    if resolved.get('entities', {}).get('distributor'):
        distributor_data = resolved['entities']['distributor']
        if isinstance(distributor_data, dict):
            distributor_value = distributor_data.get('value')
            distributor_column = distributor_data.get('column', 'distributor_name')
            if distributor_value:
                entity_filters.append(
                    f"MUST filter: {distributor_column} = '{distributor_value}' (exact match, NOT LIKE)"
                )
    
    # Superstockist filter
    if resolved.get('entities', {}).get('superstockist'):
        superstockist_data = resolved['entities']['superstockist']
        if isinstance(superstockist_data, dict):
            superstockist_value = superstockist_data.get('value')
            superstockist_column = superstockist_data.get('column')
            if superstockist_value and superstockist_column:
                entity_filters.append(
                    f"MUST filter: {superstockist_column} = '{superstockist_value}' (exact match, NOT LIKE)"
                )
    
    # Build entity filter section
    entity_filter_text = ""
    if entity_filters:
        entity_filter_text = "\n### REQUIRED FILTERS (Use exact values):\n" + "\n".join(f"- {f}" for f in entity_filters)
    
    # Time filters
    time_filter_text = ""
    if resolved.get('filters', {}).get('time_range'):
        time_range = resolved['filters']['time_range']
        time_filter_text = f"\n### TIME FILTER:\n- Use date range: {time_range[0]} to {time_range[1]}"
    
    # Determine what to SELECT based on query intent
    select_guidance = ""
    if any(metric in user_query.lower() for metric in ['sales', 'revenue', 'quantity', 'total', 'sum']):
        select_guidance = "\n### AGGREGATION REQUIRED:\n- Include SUM() or COUNT() for sales metrics\n- Use GROUP BY if needed"
    
    query_input = f"""
Generate PostgreSQL query for this request.

### User Query: 
{user_query}

### Query Context:
- Intent: {resolved.get('intent')}
- Target Table: {resolved.get('table')}
- Selected Columns: {resolved.get('columns')}
{entity_filter_text}
{time_filter_text}
{select_guidance}

### Database Schema:
{annotated_schema}

### Table Relationships:
{relationships}

### CRITICAL INSTRUCTIONS:
1. Use the EXACT product/distributor values from REQUIRED FILTERS above
2. Do NOT use LIKE, ILIKE, or wildcards - use exact equality (=)
3. Include all required filters in WHERE clause
4. Use only tables and columns from the schema
5. Apply correct JOINs from relationships if needed
6. Return ONLY the SQL query in a ```sql block
"""

    # Execute the SQL agent
    result = sql_agent_executor.invoke({
        "input": query_input,
        "annotated_schema": annotated_schema,
        "relationships": relationships
    })

    # Extract SQL query from result
    sql_query = result.get("output", "")
    
    # Clean up SQL from markdown blocks
    if "```sql" in sql_query:
        sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
    elif "```" in sql_query:
        sql_query = sql_query.split("```")[1].split("```")[0].strip()
    
    # Remove any remaining non-SQL text
    lines = sql_query.split('\n')
    sql_lines = []
    for line in lines:
        # Skip comment lines or explanatory text
        if line.strip() and not line.strip().startswith('--') and not line.strip().startswith('Note:'):
            sql_lines.append(line)
    
    sql_query = '\n'.join(sql_lines).strip()
    
    print(f"\n📝 Generated SQL:\n{sql_query}\n")
    
    return {"sql_result": sql_query}