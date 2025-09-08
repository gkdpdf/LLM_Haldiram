import streamlit as st
import psycopg2
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os 
import threading
import queue
import time
from nodes.entity_clarity_node import entity_resolver_node, load_table_columns_pg, build_catalog
from nodes.sql_creator_node import sql_agent_node
from nodes.validator_sql import validator_agent
from nodes.question_clean import question_validator
from nodes.summarized_result import summarized_results_node
from nodes.executor_sql import sql_executor_node

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SQL Query Assistant",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'waiting_for_input' not in st.session_state:
    st.session_state.waiting_for_input = False
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'current_options' not in st.session_state:
    st.session_state.current_options = []
if 'user_response' not in st.session_state:
    st.session_state.user_response = None
if 'query_result' not in st.session_state:
    st.session_state.query_result = None
if 'show_sql' not in st.session_state:
    st.session_state.show_sql = False
if 'show_details' not in st.session_state:
    st.session_state.show_details = False

# Database connection
@st.cache_resource
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        dbname="haldiram",
        user="postgres",
        password="12345678"
    )

# Graph State
class GraphState(TypedDict, total=False):
    user_query: str
    catalog: Dict[str, Any]              
    table_columns: Dict[str, List[str]]  
    annotated_schema: str
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

# Custom input function for Streamlit
def streamlit_input(prompt):
    print(f"Human input needed: {prompt}")
    
    # Extract options from prompt
    lines = prompt.split('\n')
    options = []
    for line in lines:
        if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.')):
            options.append(line.strip())
    
    # Store in session state
    st.session_state.current_question = prompt
    st.session_state.current_options = options
    st.session_state.waiting_for_input = True
    st.session_state.user_response = None
    
    # Force a rerun to show the input interface
    st.rerun()

# Override the built-in input function
import builtins
builtins.input = streamlit_input

# Build the graph
@st.cache_resource
def create_graph():
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("question_validator", question_validator)
    graph.add_node("entity_resolver", entity_resolver_node)
    graph.add_node("sql_generator", sql_agent_node)
    graph.add_node("validator_sql", validator_agent)
    graph.add_node('executor_sql', sql_executor_node)
    graph.add_node("summarized_results", summarized_results_node)
    
    # Set entry point
    graph.set_entry_point("question_validator")
    
    # Routing function
    def route_question(state):
        if state.get("route_decision") == "entity_resolver":
            return "entity_resolver"
        else:
            return "summarized_results"
    
    # Add edges
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
    graph.add_edge("summarized_results", END)
    
    return graph.compile()

# Initialize components
@st.cache_resource
def initialize_system():
    print("Initializing SQL Query Assistant...")
    conn = get_db_connection()
    table_columns = load_table_columns_pg(conn, ["tbl_shipment", "tbl_primary", "tbl_product_master"])
    catalog = build_catalog(conn, table_columns)
    compiled_graph = create_graph()
    conn.close()
    print("System initialized successfully!")
    return table_columns, catalog, compiled_graph

# Get initialized components
table_columns, catalog, compiled_graph = initialize_system()

def create_detailed_info(result):
    """Create detailed processing information"""
    details = []
    
    validation_status = result.get('validation_status', 'unknown')
    if validation_status == 'valid':
        details.append("✅ Validation: SQL query validated successfully")
    elif validation_status == 'corrected':
        details.append("🔧 Validation: SQL query was corrected automatically")
    else:
        details.append(f"❌ Validation: {validation_status}")
    
    execution_status = result.get('execution_status', 'unknown')
    if execution_status == 'success':
        execution_result = result.get('execution_result', [])
        if isinstance(execution_result, list):
            details.append(f"✅ Execution: Successfully retrieved {len(execution_result)} records")
        else:
            details.append("✅ Execution: Query executed successfully")
    else:
        execution_error = result.get('execution_error', 'Unknown error')
        details.append(f"❌ Execution: {execution_error}")
    
    route = result.get('route_decision', 'unknown')
    details.append(f"🎯 Route: {route}")
    
    resolved = result.get('resolved', {})
    if resolved:
        intent = resolved.get('intent', 'Not identified')
        entities = resolved.get('entities', [])
        details.append(f"🧠 Intent: {intent}")
        if entities:
            details.append(f"🏷️ Entities: {', '.join(entities)}")
    
    return "\n".join(details)

def process_query_async(user_query):
    """Process query in the background"""
    try:
        # Reset state
        st.session_state.waiting_for_input = False
        st.session_state.current_question = ""
        st.session_state.current_options = []
        
        # Get fresh database connection
        conn = get_db_connection()
        fresh_table_columns = load_table_columns_pg(conn, ["tbl_shipment", "tbl_primary", "tbl_product_master"])
        fresh_catalog = build_catalog(conn, fresh_table_columns)
        conn.close()
        
        # Process the query
        result = compiled_graph.invoke({
            "user_query": user_query,
            "catalog": fresh_catalog,
            "table_columns": fresh_table_columns
        })
        
        st.session_state.query_result = result
        return result
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        st.session_state.query_result = {"error": error_msg}
        return {"error": error_msg}

# Sample queries
sample_queries = [
    "How many total sales in the last month?",
    "Show me all products with Bhujia", 
    "Sales of Delhi in last 3 months",
    "Takatak sales in last two months",
    "How many distributors sold more than 5 distinct products?",
    "What are the top selling products?"
]

# Main App
def main():
    # Header
    st.title("🗄️ SQL Query Assistant")
    st.markdown("### Ask questions about your data in natural language!")
    st.markdown("This assistant helps you query your Haldiram database using plain English. Just type your question and get instant answers!")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 How to Use")
        st.markdown("""
        1. **View Results**: Answers appear in the main area
        2. **Clarification**: If needed, assistant will ask for help
        3. **Ask Questions**: Type your question at the bottom
        4. **Options**: Use checkboxes to control display
        """)
        
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        - Be specific with dates and locations
        - Use product names as they appear in database
        - Ask about sales, products, distributors, etc.
        - Try sample questions for ideas
        """)
        
        # Display options
        st.markdown("### ⚙️ Display Options")
        st.session_state.show_sql = st.checkbox("Show Generated SQL", value=st.session_state.show_sql)
        st.session_state.show_details = st.checkbox("Show Processing Details", value=st.session_state.show_details)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Results Section
        st.markdown("### 📊 Results")
        
        if st.session_state.query_result:
            result = st.session_state.query_result
            
            if "error" in result:
                st.error(result["error"])
            else:
                # Main answer
                final_answer = result.get('final_output', 'No result generated.')
                st.success(final_answer)
                
                # SQL output if requested
                if st.session_state.show_sql:
                    generated_sql = result.get('validated_sql', 'No SQL generated.')
                    if generated_sql != 'No SQL generated.':
                        st.markdown("### 🔧 Generated SQL")
                        st.code(generated_sql, language='sql')
                
                # Processing details if requested
                if st.session_state.show_details:
                    st.markdown("### 🔍 Processing Details")
                    details = create_detailed_info(result)
                    st.text(details)
        else:
            st.info("👆 Ask a question below to get started!")
        
        # Human Input Section (when needed)
        if st.session_state.waiting_for_input:
            st.markdown("---")
            st.markdown("### 🤔 Assistant Needs Help")
            
            st.warning("I need your help to clarify something...")
            
            # Show the question
            st.text_area(
                "Question from Assistant:",
                value=st.session_state.current_question,
                height=150,
                disabled=True
            )
            
            # Show options
            if st.session_state.current_options:
                choice = st.radio(
                    "Please select your choice:",
                    options=st.session_state.current_options,
                    key="user_choice_radio"
                )
                
                if st.button("Submit My Choice", type="primary"):
                    # Extract number from choice
                    if choice and choice.startswith(('1.', '2.', '3.', '4.', '5.', '6.')):
                        choice_num = choice.split('.')[0]
                        st.session_state.user_response = choice_num
                        st.session_state.waiting_for_input = False
                        st.success("Choice submitted! Processing continues...")
                        st.rerun()
            else:
                st.warning("Waiting for options...")
        
        # User Input Section (at the bottom)
        st.markdown("---")
        st.markdown("### ❓ Ask Your Question")
        
        # Sample questions
        with st.expander("📝 Sample Questions"):
            st.markdown("Click on any sample question to try it:")
            for i, query in enumerate(sample_queries):
                if st.button(f"{query}", key=f"sample_{i}"):
                    st.session_state.user_query_input = query
                    st.rerun()
        
        # Main input
        user_query = st.text_area(
            "Type your question here:",
            placeholder="e.g., How many sales in Delhi last month?",
            height=100,
            key="user_query_input"
        )
        
        # Submit button
        col_btn1, col_btn2 = st.columns([3, 1])
        
        with col_btn1:
            if st.button("🚀 Get Answer", type="primary", use_container_width=True):
                if user_query.strip():
                    # Check if we're waiting for human input
                    if st.session_state.waiting_for_input:
                        st.warning("Please respond to the assistant's question above first.")
                    else:
                        # Process the query
                        with st.spinner("Processing your query..."):
                            try:
                                result = process_query_async(user_query)
                                if result and "error" not in result:
                                    st.success("Query processed successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please enter a question.")
        
        with col_btn2:
            if st.button("🔄 Clear Results", use_container_width=True):
                st.session_state.query_result = None
                st.session_state.waiting_for_input = False
                st.session_state.current_question = ""
                st.session_state.current_options = []
                st.rerun()
    
    with col2:
        # Status indicator
        if st.session_state.waiting_for_input:
            st.error("🔴 Waiting for user input")
        elif st.session_state.query_result:
            st.success("🟢 Ready")
        else:
            st.info("🔵 Ready to query")
        
        # Quick stats or additional info can go here
        st.markdown("### 📈 Quick Stats")
        st.markdown("- Database: Haldiram")
        st.markdown("- Tables: 3 main tables")
        st.markdown("- Status: Connected")

if __name__ == "__main__":
    main()