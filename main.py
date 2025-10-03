from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import os 
from nodes.entity_clarity_node import entity_resolver_node, load_table_columns_pg, build_catalog
from nodes.sql_creator_node import sql_agent_node
from nodes.validator_sql import validator_agent
from nodes.question_clean import question_validator
from nodes.summarized_result import summarized_results_node
from nodes.executor_sql import sql_executor_node
from nodes.chart_creation_node import chart_creation_node
import psycopg2
import asyncio
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

load_dotenv()

with open("annotated_schema.md", "r", encoding="utf-8") as f:
    annotated_schema = f.read()

with open("relationship.txt", "r", encoding="utf-8") as f:
    relationships = f.read()

# PostgreSQL connection for data queries (psycopg2)
conn = psycopg2.connect(
    host="localhost",
    dbname="haldiram",
    user="postgres",
    password="12345678"
)

# ---------- Setup Async PostgreSQL Memory ----------
DB_URI = "postgresql://postgres:12345678@localhost/haldiram"

async def setup_memory():
    """Setup async PostgreSQL checkpointer and conversation history"""
    try:
        # Create async connection pool
        pool = AsyncConnectionPool(
            conninfo=DB_URI,
            max_size=10,
            kwargs={"autocommit": True, "row_factory": dict_row}
        )
        
        # Create checkpointer
        checkpointer = AsyncPostgresSaver(pool)
        
        # Setup checkpoint tables
        await checkpointer.setup()
        
        print("✓ Async PostgreSQL checkpointer initialized")
        
        # Setup conversation history table
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_history (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) NOT NULL,
                        user_query TEXT NOT NULL,
                        final_output TEXT,
                        validated_sql TEXT,
                        resolved_entities JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                await cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_id 
                    ON conversation_history(session_id);
                """)
                
                await cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_created_at 
                    ON conversation_history(created_at);
                """)
        
        print("✓ Conversation history table ready")
        
        return checkpointer, pool
        
    except Exception as e:
        print(f"Error setting up memory: {e}")
        raise

async def get_conversation_history(pool, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve conversation history from PostgreSQL"""
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT user_query, final_output, resolved_entities, created_at 
                    FROM conversation_history 
                    WHERE session_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (session_id, limit))
                
                rows = await cur.fetchall()
                
                history = [
                    {
                        "user_query": row["user_query"],
                        "response": row["final_output"],
                        "entities": row["resolved_entities"],
                        "timestamp": row["created_at"].isoformat() if row["created_at"] else None
                    }
                    for row in reversed(rows)
                ]
                
                return history
    except Exception as e:
        print(f"Error retrieving history: {e}")
        return []

async def save_conversation(pool, session_id: str, user_query: str, final_output: str, 
                           validated_sql: Optional[str] = None, resolved_entities: Optional[Dict] = None):
    """Save conversation to PostgreSQL"""
    try:
        import json
        entities_json = json.dumps(resolved_entities) if resolved_entities else None
        
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO conversation_history 
                    (session_id, user_query, final_output, validated_sql, resolved_entities)
                    VALUES (%s, %s, %s, %s, %s)
                """, (session_id, user_query, final_output, validated_sql, entities_json))
                
    except Exception as e:
        print(f"Error saving conversation: {e}")

# ---------- Graph State ----------
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
    # Memory fields
    session_id: Optional[str]
    conversation_history: Optional[List[Dict[str, Any]]]
    last_entities: Optional[Dict[str, Any]]

# ---------- Build Graph ----------
graph = StateGraph(GraphState)

# Add all nodes
graph.add_node("question_validator", question_validator)
graph.add_node("entity_resolver", entity_resolver_node)
graph.add_node("sql_generator", sql_agent_node)
graph.add_node("validator_sql", validator_agent)
graph.add_node('executor_sql', sql_executor_node)
graph.add_node("summarized_results", summarized_results_node)
graph.add_node("chart_creation_node", chart_creation_node)

graph.set_entry_point("question_validator")

def route_question(state):
    if state.get("route_decision") == "entity_resolver":
        return "entity_resolver"
    else:
        return "summarized_results"

graph.add_conditional_edges(
    "question_validator",
    route_question,
    {
        "entity_resolver": "entity_resolver",
        "summarized_results": "summarized_results"
    }
)

graph.add_edge("entity_resolver", "sql_generator")
graph.add_edge("sql_generator", "validator_sql")
graph.add_edge("validator_sql", "executor_sql") 
graph.add_edge("executor_sql", "summarized_results") 
graph.add_edge("summarized_results", "chart_creation_node")
graph.add_edge("chart_creation_node", END)

# ---------- Main Async Execution ----------

async def main():
    """Main async function to run the conversation loop"""
    
    # Setup memory
    checkpointer, pool = await setup_memory()
    
    # Compile graph with async checkpointer
    compiled = graph.compile(checkpointer=checkpointer)
    print("✓ Graph compiled with async memory support")
    
    # Load table metadata
    table_columns = load_table_columns_pg(conn, ["tbl_shipment", "tbl_primary", "tbl_product_master"])
    catalog = build_catalog(conn, table_columns)
    
    # Session ID
    SESSION_ID = "user_123_session_001"
    
    # Configuration for memory
    config = {
        "configurable": {
            "thread_id": SESSION_ID,
        }
    }
    
    print("\n" + "="*60)
    print("Welcome to Haldiram Analytics Assistant!")
    print("="*60)
    print("Ask questions about your data. Type 'exit' or 'quit' to end.\n")
    
    # Track last resolved entities
    last_resolved_entities = {}
    
    # Continuous conversation loop
    while True:
        try:
            # Get user input
            user_query = await asyncio.to_thread(input, "\nYou: ")
            user_query = user_query.strip()
            
            # Exit conditions
            if user_query.lower() in ['exit', 'quit', 'q', 'bye']:
                print("\nGoodbye! Your conversation history has been saved.")
                break
            
            # Skip empty queries
            if not user_query:
                continue
            
            # Retrieve conversation history
            conversation_history = await get_conversation_history(pool, SESSION_ID, limit=5)
            
            # Check if this is a reference question about previous context
            reference_keywords = [
                'which', 'what was', 'last', 'previous', 'earlier', 
                'that product', 'that distributor', 'which one',
                'what did i ask', 'my last', 'my previous'
            ]
            
            is_reference = any(keyword in user_query.lower() for keyword in reference_keywords)
            
            # Handle reference questions directly using LLM
            if is_reference and (conversation_history or last_resolved_entities):
                print("\n📝 Checking conversation context...")
                
                # Build context summary
                context_parts = []
                
                if last_resolved_entities:
                    if last_resolved_entities.get('product_name'):
                        context_parts.append(f"Product: {last_resolved_entities['product_name']}")
                    if last_resolved_entities.get('distributor_name'):
                        context_parts.append(f"Distributor: {last_resolved_entities['distributor_name']}")
                    if last_resolved_entities.get('material_description'):
                        context_parts.append(f"Material: {last_resolved_entities['material_description']}")
                
                if conversation_history:
                    recent_query = conversation_history[-1].get('user_query', '')
                    context_parts.append(f"Last question: {recent_query}")
                
                if context_parts:
                    # Use simple LLM to answer from context
                    from langchain_openai import ChatOpenAI
                    from langchain_core.prompts import ChatPromptTemplate
                    
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful assistant. Answer the user's question based on the context provided. Be brief and direct."),
                        ("user", "Context: {context}\n\nQuestion: {question}\n\nAnswer:")
                    ])
                    
                    response = llm.invoke(
                        prompt.format_messages(
                            context="\n".join(context_parts),
                            question=user_query
                        )
                    )
                    
                    # Display answer
                    print("\n" + "-"*60)
                    print("Assistant:")
                    print(response.content)
                    print("-"*60)
                    
                    # Save this interaction
                    await save_conversation(
                        pool=pool,
                        session_id=SESSION_ID,
                        user_query=user_query,
                        final_output=response.content,
                        validated_sql=None,
                        resolved_entities=last_resolved_entities
                    )
                    
                    continue  # Skip graph execution for reference questions
            
            # Build context for non-reference queries
            context_info = ""
            if last_resolved_entities:
                entity_list = [f"{k}: {v}" for k, v in last_resolved_entities.items()]
                if entity_list:
                    context_info = f"\n\nPrevious entities: {', '.join(entity_list)}"
            
            print("\nProcessing your query...")
            
            # Invoke graph asynchronously
            result = await compiled.ainvoke(
                {
                    "user_query": user_query + context_info,
                    "catalog": catalog,             
                    "table_columns": table_columns,
                    "annotated_schema": annotated_schema,
                    "relationships": relationships,
                    "session_id": SESSION_ID,
                    "conversation_history": conversation_history,
                    "last_entities": last_resolved_entities
                },
                config=config
            )
            
            # Update last resolved entities
            if result.get("resolved"):
                last_resolved_entities.update(result.get("resolved", {}))
            
            # Save conversation
            await save_conversation(
                pool=pool,
                session_id=SESSION_ID,
                user_query=result.get("user_query", ""),
                final_output=result.get("final_output", ""),
                validated_sql=result.get("validated_sql"),
                resolved_entities=result.get("resolved")
            )
            
            # Display result
            print("\n" + "-"*60)
            print("Assistant:")
            print(result['final_output'])
            print("-"*60)
            
            # Show SQL if successful
            if result.get("validated_sql") and result.get("execution_status") == "success":
                sql_preview = result.get('validated_sql', '')[:120]
                print(f"\n[SQL: {sql_preview}...]")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            print("Please try again or type 'exit' to quit.")
    
    # Cleanup
    await pool.close()
    conn.close()
    print("\nConnections closed. Session saved.")

# Run the async main function
if __name__ == "__main__":
    import sys
    import selectors
    
    # Fix for Windows: Use SelectorEventLoop instead of ProactorEventLoop
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())