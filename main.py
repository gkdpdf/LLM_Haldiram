from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
import psycopg2
import os

load_dotenv()

with open("annotated_schema.md", "r", encoding="utf-8") as f:
    annotated_schema = f.read()

with open("relationship.txt", "r", encoding="utf-8") as f:
    relationships = f.read()

conn = psycopg2.connect(
    host="localhost",
    dbname="haldiram",
    user="postgres",
    password="12345678"
)

# Import after connection setup
from nodes.entity_clarity_node import load_table_columns_pg, build_catalog
from nodes.sql_creator_node import sql_agent_node
from nodes.validator_sql import validator_agent
from nodes.question_clean import question_validator
from nodes.summarized_result import summarized_results_node
from nodes.executor_sql import sql_executor_node
from nodes.chart_creation_node import chart_creation_node

# ---------- Enhanced Graph State with Memory ----------
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
    session_entities: Dict[str, Any]
    last_product: Optional[str]
    last_distributor: Optional[str]
    last_table: Optional[str]


# ---------- Enhanced Entity Resolver with Memory ----------
def entity_resolver_with_memory(state: GraphState):
    """
    Enhanced entity resolver with smart context reuse using MemorySaver
    """
    from nodes.entity_clarity_node import detect_time_filters, llm_understand
    
    user_query = state["user_query"]
    catalog = state.get("catalog", {})
    table_columns = state.get("table_columns", {})
    session_entities = state.get("session_entities", {})
    
    print("\n🧠 Checking session memory for previous entities...")
    if session_entities:
        print(f"📝 Session memory: {list(session_entities.keys())}")
        if 'product' in session_entities:
            val = session_entities['product'].get('value', session_entities['product']) if isinstance(session_entities['product'], dict) else session_entities['product']
            print(f"    - product: {val}")
    
    # PRE-CHECK: Detect continuation BEFORE calling resolver
    query_lower = user_query.lower()
    
    # Reference keywords
    reference_keywords = ['last', 'previous', 'same', 'that', 'it', 'this']
    has_reference = any(kw in query_lower for kw in reference_keywords)
    
    # Check for sales/time queries without specific entity names
    has_metric = any(m in query_lower for m in ['sales', 'revenue', 'total', 'sum', 'top', 'rank'])
    has_time = any(t in query_lower for t in ['may', 'june', 'january', 'february', 'march', 'april', 
                                                'july', 'august', 'september', 'october', 'november', 'december',
                                                'month', 'year', 'week', 'day', 'today', 'yesterday', 'last'])
    mentions_entity = any(e in query_lower for e in ['bhujia', 'namkeen', 'sev', 'gathiya', 'mixture', 'chips', 'papad', 
                      'trading', 'company', 'distributor'])
    
    # CONTINUATION: metric/time query + product in memory + no new entity name
    is_continuation = (
        session_entities and
        'product' in session_entities and
        (has_metric or has_time) and
        not mentions_entity
    )
    
    # SKIP RESOLVER if continuation or reference
    if is_continuation or has_reference:
        print("✅ CONTINUATION - Skipping prompts, reusing session")
        
        filters = detect_time_filters(user_query)
        parsed = llm_understand(user_query)
        
        resolved = {
            "intent": parsed.get("intent", "query"),
            "metrics": parsed.get("metrics", []),
            "entities": {},
            "filters": filters,
            "table": session_entities.get("table"),
            "columns": session_entities.get("columns", [])
        }
        
        # Reuse all entities
        for entity_type in ["product", "distributor", "superstockist"]:
            if entity_type in session_entities:
                resolved["entities"][entity_type] = session_entities[entity_type]
                val = session_entities[entity_type].get('value', 'N/A') if isinstance(session_entities[entity_type], dict) else session_entities[entity_type]
                print(f"  ♻️ Reusing {entity_type}: {val}")
        
        updated_session_entities = session_entities.copy()
        
    else:
        # NEW QUERY - call resolver with prompts
        print("🆕 New query - running resolver")
        
        from nodes.entity_clarity_node import resolve_with_human_in_loop_pg
        
        try:
            resolved = resolve_with_human_in_loop_pg(user_query, catalog, table_columns)
        except Exception as e:
            print(f"⚠️ Error: {e}")
            resolved = {
                "intent": "query",
                "entities": {},
                "filters": {},
                "table": session_entities.get("table"),
                "columns": session_entities.get("columns", [])
            }
        
        if not isinstance(resolved.get("entities"), dict):
            resolved["entities"] = {}
        
        updated_session_entities = session_entities.copy()
        
        # Save new entities
        if resolved.get("entities"):
            for entity_type, entity_data in resolved["entities"].items():
                if entity_data:
                    updated_session_entities[entity_type] = entity_data
                    val = entity_data.get('value', entity_data) if isinstance(entity_data, dict) else entity_data
                    print(f"💾 Storing {entity_type}: {val}")
        
        if resolved.get("table"):
            updated_session_entities["table"] = resolved["table"]
            print(f"💾 Storing table: {resolved['table']}")
        
        if resolved.get("columns"):
            updated_session_entities["columns"] = resolved["columns"]
            print(f"💾 Storing columns: {resolved['columns']}")
    
    try:
        with open("annotated_schema.md", "r", encoding="utf-8") as f:
            annotated_schema = f.read()
    except FileNotFoundError:
        annotated_schema = "Schema not found"
    
    print(f"\n✅ Final state:")
    print(f"   Entities: {list(resolved.get('entities', {}).keys())}")
    if resolved.get('entities', {}).get('product'):
        pval = resolved['entities']['product'].get('value', 'N/A') if isinstance(resolved['entities']['product'], dict) else resolved['entities']['product']
        print(f"   Product: {pval}")
    print(f"   Filters: {resolved.get('filters', {})}")
    
    return {
        "resolved": resolved,
        "annotated_schema": annotated_schema,
        "session_entities": updated_session_entities,
        "last_product": resolved.get("entities", {}).get("product", {}).get("value") if isinstance(resolved.get("entities", {}).get("product"), dict) else None,
        "last_distributor": resolved.get("entities", {}).get("distributor", {}).get("value") if isinstance(resolved.get("entities", {}).get("distributor"), dict) else None,
        "last_table": resolved.get("table")
    }


# ---------- Build Graph with Memory ----------
def create_graph_with_memory():
    """Create the LangGraph with MemorySaver for session persistence"""
    
    graph = StateGraph(GraphState)
    
    # Add all nodes
    graph.add_node("question_validator", question_validator)
    graph.add_node("entity_resolver", entity_resolver_with_memory)
    graph.add_node("sql_generator", sql_agent_node)
    graph.add_node("validator_sql", validator_agent)
    graph.add_node('executor_sql', sql_executor_node)
    graph.add_node("summarized_results", summarized_results_node)
    graph.add_node("chart_creation_node", chart_creation_node)
    
    # Set entry point
    graph.set_entry_point("question_validator")
    
    # Routing logic
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
    
    # Add edges
    graph.add_edge("entity_resolver", "sql_generator")
    graph.add_edge("sql_generator", "validator_sql")
    graph.add_edge("validator_sql", "executor_sql")
    graph.add_edge("executor_sql", "summarized_results")
    graph.add_edge("summarized_results", "chart_creation_node")
    graph.add_edge("chart_creation_node", END)
    
    # Compile with MemorySaver
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)
    
    return compiled


# ---------- Session Management ----------
class SessionManager:
    """Manage conversation sessions with persistent memory"""
    
    def __init__(self, graph, catalog, table_columns, annotated_schema, relationships):
        self.graph = graph
        self.catalog = catalog
        self.table_columns = table_columns
        self.annotated_schema = annotated_schema
        self.relationships = relationships
        self.sessions = {}
    
    def create_session(self, session_id: str):
        """Create a new session"""
        self.sessions[session_id] = {
            "created_at": __import__('datetime').datetime.now(),
            "query_count": 0
        }
        print(f"\n🆕 Created new session: {session_id}")
    
    def invoke(self, user_query: str, session_id: str = "default"):
        """Invoke graph with session memory"""
        
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        self.sessions[session_id]["query_count"] += 1
        
        print(f"\n{'='*70}")
        print(f"📊 Session: {session_id} | Query #{self.sessions[session_id]['query_count']}")
        print(f"{'='*70}")
        
        config = {"configurable": {"thread_id": session_id}}
        
        # Let MemorySaver load session_entities automatically
        result = self.graph.invoke(
            {
                "user_query": user_query,
                "catalog": self.catalog,
                "table_columns": self.table_columns,
                "annotated_schema": self.annotated_schema,
                "relationships": self.relationships
            },
            config=config
        )
        
        return result
    
    def get_session_history(self, session_id: str):
        """Get entity history for a session"""
        if session_id in self.sessions:
            return self.sessions[session_id]
        return None
    
    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            print(f"🗑️ Cleared session: {session_id}")


# ---------- Interactive Loop ----------
if __name__ == "__main__":
    # Load tables and catalog
    table_columns = load_table_columns_pg(
        conn, 
        ["tbl_shipment", "tbl_primary", "tbl_product_master"]
    )
    catalog = build_catalog(conn, table_columns)
    
    # Create graph with memory
    graph = create_graph_with_memory()
    
    # Create session manager
    session_manager = SessionManager(
        graph=graph,
        catalog=catalog,
        table_columns=table_columns,
        annotated_schema=annotated_schema,
        relationships=relationships
    )
    
    # Interactive conversation loop
    print("\n" + "="*70)
    print("Text-to-SQL Interactive System with Memory")
    print("="*70)
    print("Commands:")
    print("  - Type your question to query the database")
    print("  - 'new session' - Start a new conversation session")
    print("  - 'exit' - Quit the program")
    print("="*70)
    
    session_id = "default"
    query_number = 0
    
    while True:
        print(f"\nSession: {session_id} | Query #{query_number + 1}")
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'exit':
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'new session':
            session_id = input("Enter new session ID (or press Enter for auto-generated): ").strip()
            if not session_id:
                import uuid
                session_id = str(uuid.uuid4())[:8]
            query_number = 0
            print(f"Started new session: {session_id}")
            continue
        
        query_number += 1
        
        try:
            result = session_manager.invoke(
                user_query=user_input,
                session_id=session_id
            )
            
            print("\n" + "-"*70)
            print("RESULT:")
            print("-"*70)
            print(result.get('final_output', 'No output generated'))
            print("-"*70)
            
        except Exception as e:
            print(f"\nError processing query: {e}")
            import traceback
            traceback.print_exc()