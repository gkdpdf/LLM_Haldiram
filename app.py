import gradio as gr
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
import os 
import psycopg2
import sys
import io
import threading
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from contextlib import redirect_stdout, redirect_stderr
from nodes.entity_clarity_node import entity_resolver_node, load_table_columns_pg, build_catalog
from nodes.sql_creator_node import sql_agent_node
from nodes.validator_sql import validator_agent
from nodes.question_clean import question_validator
from nodes.summarized_result import summarized_results_node
from nodes.chart_creation_node import chart_creation_node
from nodes.executor_sql import sql_executor_node

load_dotenv()

# Global state for UI interaction
class UIState:
    def __init__(self):
        self.waiting_for_input = False
        self.current_question = ""
        self.user_response = None
        self.response_ready = threading.Event()
        self.graph_state = None
        self.current_node = None
        self.captured_output = ""
        self.processing_complete = False
        self.final_result = None
        self.chart_data = None
        self.chart_figure = None
        
ui_state = UIState()

class OutputCapture:
    def __init__(self):
        self.buffer = []
        
    def write(self, text):
        # Store the output
        self.buffer.append(text)
        ui_state.captured_output += text
        # Also print to original stdout so you can see it in terminal
        sys.__stdout__.write(text)
        
    def flush(self):
        sys.__stdout__.flush()

output_capture = OutputCapture()

def ui_input(prompt):
    """Modified input function that works with Gradio UI"""
    # Add the prompt to captured output so it shows in chat
    full_prompt = f"\n{prompt}"
    ui_state.captured_output += full_prompt
    print(full_prompt)  # This will also be captured
    
    # Set the state to waiting for input
    ui_state.current_question = prompt
    ui_state.waiting_for_input = True
    ui_state.response_ready.clear()
    
    # Wait for user response (with timeout to avoid hanging)
    if ui_state.response_ready.wait(timeout=300):  # 5 minute timeout
        response = ui_state.user_response
        ui_state.user_response = None
        ui_state.waiting_for_input = False
        
        # Add the user's response to captured output
        ui_state.captured_output += f"\n✅ Your response: {response}\n"
        return response
    else:
        ui_state.waiting_for_input = False
        return "No response provided"

# Override the built-in input function
import builtins
builtins.input = ui_input

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        dbname="haldiram",
        user="postgres",
        password="12345678"
    )

# Chart generation with comprehensive rules
class ChartRules:
    @staticmethod
    def should_create_chart(df, query):
        """Determine if a chart should be created based on data and query"""
        if df is None or df.empty:
            return False
        
        # Must have at least 2 rows for meaningful visualization
        if len(df) < 2:
            return False
        
        # Must have at least one numeric column
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if len(numeric_cols) == 0:
            return False
        
        # Check query for chart-worthy keywords
        query_lower = query.lower()
        chart_keywords = [
            'sales', 'revenue', 'total', 'sum', 'count', 'amount', 'value',
            'trend', 'analysis', 'compare', 'comparison', 'by', 'over', 
            'top', 'bottom', 'highest', 'lowest', 'distribution', 'breakdown',
            'monthly', 'daily', 'yearly', 'timeline', 'progress'
        ]
        
        return any(keyword in query_lower for keyword in chart_keywords)
    
    @staticmethod
    def detect_chart_type(df, query):
        """Smart chart type detection based on data structure and query intent"""
        if df is None or df.empty:
            return None
        
        query_lower = query.lower()
        
        # Time series indicators
        time_keywords = ['trend', 'over time', 'monthly', 'daily', 'yearly', 'timeline', 'progression', 'growth']
        if any(keyword in query_lower for keyword in time_keywords):
            return 'line'
        
        # Distribution/percentage indicators  
        dist_keywords = ['distribution', 'breakdown', 'percentage', 'share', 'proportion', 'composition']
        if any(keyword in query_lower for keyword in dist_keywords):
            return 'pie'
        
        # Comparison indicators
        comp_keywords = ['compare', 'comparison', 'vs', 'versus', 'by', 'each', 'per', 'top', 'bottom', 'rank']
        if any(keyword in query_lower for keyword in comp_keywords):
            return 'bar'
        
        # Data structure based detection
        date_like_cols = []
        for col in df.columns:
            col_name = str(col).lower()
            if any(date_word in col_name for date_word in ['date', 'time', 'month', 'year', 'day']):
                date_like_cols.append(col)
        
        if date_like_cols:
            return 'line'  # Time series data
        
        # Default based on row count
        if len(df) <= 20:
            return 'bar'  # Good for categorical comparisons
        else:
            return 'line'  # Better for large datasets

def create_chart(df, query):
    """Create chart with comprehensive rules and error handling"""
    try:
        # Validate inputs
        if not ChartRules.should_create_chart(df, query):
            return None
        
        # Clean and prepare data
        df_clean = df.copy()
        df_clean = df_clean.dropna()
        
        if df_clean.empty:
            return None
        
        # Limit data for performance
        if len(df_clean) > 100:
            df_clean = df_clean.head(100)
        
        # Detect chart type
        chart_type = ChartRules.detect_chart_type(df_clean, query)
        if not chart_type:
            return None
        
        # Identify columns
        numeric_cols = [col for col in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[col])]
        text_cols = [col for col in df_clean.columns if col not in numeric_cols]
        
        # Select appropriate columns
        if len(text_cols) > 0:
            x_col = text_cols[0]
        else:
            x_col = df_clean.columns[0]
        
        if len(numeric_cols) > 0:
            y_col = numeric_cols[0]
        else:
            y_col = df_clean.columns[1] if len(df_clean.columns) > 1 else df_clean.columns[0]
        
        # Ensure different columns
        if x_col == y_col and len(df_clean.columns) > 1:
            cols = list(df_clean.columns)
            x_col = cols[0]
            y_col = cols[1]
        
        # Create the chart
        fig = None
        title = f"{query[:50]}..." if len(query) > 50 else query
        
        if chart_type == 'bar':
            # Sort data for better visualization
            try:
                df_sorted = df_clean.sort_values(by=y_col, ascending=False)
            except:
                df_sorted = df_clean
            
            fig = px.bar(
                df_sorted, 
                x=x_col, 
                y=y_col,
                title=f"Bar Chart: {title}",
                labels={x_col: x_col.title(), y_col: y_col.title()}
            )
            fig.update_layout(xaxis_tickangle=-45)
            
        elif chart_type == 'line':
            fig = px.line(
                df_clean, 
                x=x_col, 
                y=y_col,
                title=f"Trend Analysis: {title}",
                labels={x_col: x_col.title(), y_col: y_col.title()},
                markers=True
            )
            
        elif chart_type == 'pie':
            fig = px.pie(
                df_clean, 
                values=y_col, 
                names=x_col,
                title=f"Distribution: {title}"
            )
        
        if fig is None:
            return None
        
        # Apply consistent styling
        fig.update_layout(
            height=400,
            margin=dict(l=50, r=50, t=70, b=50),
            font=dict(size=11),
            template="plotly_white",
            title=dict(x=0.5, font=dict(size=14))
        )
        
        # Add hover formatting for non-pie charts
        if chart_type != 'pie':
            fig.update_traces(
                hovertemplate=f'<b>{x_col}</b>: %{{x}}<br><b>{y_col}</b>: %{{y:,.0f}}<extra></extra>'
            )
        
        return fig
        
    except Exception as e:
        print(f"Chart creation error: {str(e)}")
        return None

def process_sql_results_for_chart(execution_result, query):
    """Process SQL results and create chart if appropriate"""
    try:
        ui_state.chart_data = None
        ui_state.chart_figure = None
        
        if not execution_result or not isinstance(execution_result, list) or len(execution_result) == 0:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(execution_result)
        ui_state.chart_data = df
        
        # Try to create chart
        if ChartRules.should_create_chart(df, query):
            fig = create_chart(df, query)
            if fig:
                ui_state.chart_figure = fig
                return fig
        
        return None
        
    except Exception as e:
        print(f"Error processing SQL results for chart: {str(e)}")
        return None


with open("annotated_schema.md", "r", encoding="utf-8") as f:
    annotated_schema = f.read()

with open("relationship.txt", "r", encoding="utf-8") as f:
    relationships = f.read()

# Graph State
class GraphState(TypedDict, total=False):
    user_query: str
    catalog: Dict[str, Any]              
    table_columns: Dict[str, List[str]]  
    annotated_schema: str
    relationships:str
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

def create_graph():
    graph = StateGraph(GraphState)
    graph.add_node("question_validator", question_validator)
    graph.add_node("entity_resolver", entity_resolver_node)
    graph.add_node("sql_generator", sql_agent_node)
    graph.add_node("validator_sql", validator_agent)
    graph.add_node('executor_sql', sql_executor_node)
    graph.add_node("chart_creation_node",chart_creation_node)
    graph.add_node("summarized_results", summarized_results_node)
    
    graph.set_entry_point("question_validator")
    
    def route_question(state):
        return "entity_resolver" if state.get("route_decision") == "entity_resolver" else "summarized_results"
    
    graph.add_conditional_edges("question_validator", route_question, {
        "entity_resolver": "entity_resolver", 
        "summarized_results": "summarized_results"
    })
    
    graph.add_edge("entity_resolver", "sql_generator")
    graph.add_edge("sql_generator", "validator_sql")
    graph.add_edge("validator_sql", "executor_sql") 
    graph.add_edge("executor_sql", "summarized_results") 
    # End after summarization
    graph.add_edge("summarized_results", "chart_creation_node")

    graph.add_edge("chart_creation_node",END)
    
    return graph.compile()

# Initialize system
print("Initializing SQL Query Assistant...")
conn = get_db_connection()
try:
    table_columns = load_table_columns_pg(conn, ["tbl_shipment", "tbl_primary", "tbl_product_master"])
    catalog = build_catalog(conn, table_columns)
finally:
    conn.close()

compiled_graph = create_graph()
print("System initialized!")

def run_graph_with_capture(message):
    """Run the graph and capture all output"""
    try:
        # Reset state
        ui_state.captured_output = ""
        ui_state.processing_complete = False
        ui_state.final_result = None
        ui_state.chart_data = None
        ui_state.chart_figure = None
        
        # Get fresh connection and data
        conn = get_db_connection()
        try:
            fresh_table_columns = load_table_columns_pg(conn, ["tbl_shipment", "tbl_primary", "tbl_product_master"])
            fresh_catalog = build_catalog(conn, fresh_table_columns)
        finally:
            conn.close()
        
        # Store state for potential continuation
        ui_state.graph_state = {
            "user_query": message,
            "catalog": fresh_catalog,
            "table_columns": fresh_table_columns
        }
        
        # Capture stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            sys.stdout = output_capture
            sys.stderr = output_capture
            
            # Execute the graph
            result = compiled_graph.invoke(ui_state.graph_state)
            
            # Process results for chart creation
            if result and 'execution_result' in result:
                process_sql_results_for_chart(result['execution_result'], message)
            
            # Store the result
            ui_state.final_result = result
            ui_state.processing_complete = True
            
            return result
            
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        
    except Exception as e:
        ui_state.captured_output += f"\n❌ Error: {str(e)}"
        ui_state.processing_complete = True
        raise e

def process_query(message, history):
    """Process a new query"""
    if not message.strip():
        return history, "", None
    
    history = history or []
    history.append([message, "🔄 Processing your query..."])
    
    # Reset UI state
    ui_state.waiting_for_input = False
    ui_state.current_question = ""
    ui_state.user_response = None
    ui_state.captured_output = ""
    ui_state.processing_complete = False
    ui_state.chart_data = None
    ui_state.chart_figure = None
    
    # Start processing in a thread
    def process_thread():
        try:
            result = run_graph_with_capture(message)
        except Exception as e:
            # Error handling is done in run_graph_with_capture
            pass
    
    thread = threading.Thread(target=process_thread)
    thread.daemon = True
    thread.start()
    
    # Wait a bit and then check status
    time.sleep(1)
    
    # Check if we're waiting for input or if processing is complete
    max_wait = 30  # Maximum 30 seconds to wait for initial processing
    waited = 0
    
    while waited < max_wait and not ui_state.processing_complete and not ui_state.waiting_for_input:
        time.sleep(0.5)
        waited += 0.5
    
    if ui_state.waiting_for_input:
        # Show captured output plus the question
        response_text = f"**Processing Steps:**\n```\n{ui_state.captured_output}\n```\n\n"
        response_text += f"❓ **I need clarification:**\n\n{ui_state.current_question}\n\n"
        response_text += "👇 Please provide your response in the text box below and click 'Submit Response'."
        history[-1][1] = response_text
        return history, "", None
        
    elif ui_state.processing_complete:
        # Show final result
        response_text = f"**Processing Steps:**\n```\n{ui_state.captured_output}\n```\n\n"
        
        if ui_state.final_result:
            final_answer = ui_state.final_result.get('final_output', 'No result generated.')
            generated_sql = ui_state.final_result.get('validated_sql', '')
            
            response_text += f"**Final Answer:** {final_answer}"
            if generated_sql and generated_sql != 'No SQL generated.':
                response_text += f"\n\n**Generated SQL:**\n```sql\n{generated_sql}\n```"
        
        history[-1][1] = response_text
        
        # Return chart if available
        return history, "", ui_state.chart_figure
        
    else:
        history[-1][1] = "⏱️ Processing is taking longer than expected. Please wait or try again."
        return history, "", None

def process_user_response(user_response, history):
    """Process user response to clarification"""
    if not user_response.strip():
        return history, "", "Please provide a response before submitting.", None
    
    if not ui_state.waiting_for_input:
        return history, "", "No clarification was requested. Please ask a new question.", None
    
    # Add user's response to chat
    if history:
        history.append([f"**My response:** {user_response}", "🔄 Processing with your response..."])
    
    # Provide the response to the waiting graph
    ui_state.user_response = user_response.strip()
    ui_state.response_ready.set()
    
    # Wait for processing to complete
    max_wait = 30
    waited = 0
    
    while waited < max_wait and not ui_state.processing_complete:
        time.sleep(0.5)
        waited += 0.5
        
        # Check if another input is needed
        if ui_state.waiting_for_input and ui_state.current_question:
            response_text = f"**Processing Steps:**\n```\n{ui_state.captured_output}\n```\n\n"
            response_text += f"❓ **I need more clarification:**\n\n{ui_state.current_question}\n\n"
            response_text += "👇 Please provide your response in the text box below and click 'Submit Response'."
            history[-1][1] = response_text
            return history, "", "Another clarification needed.", None
    
    # Final result
    if ui_state.processing_complete and ui_state.final_result:
        response_text = f"**Processing Steps:**\n```\n{ui_state.captured_output}\n```\n\n"
        
        final_answer = ui_state.final_result.get('final_output', 'No result generated.')
        generated_sql = ui_state.final_result.get('validated_sql', '')
        
        response_text += f"**Final Answer:** {final_answer}"
        if generated_sql and generated_sql != 'No SQL generated.':
            response_text += f"\n\n**Generated SQL:**\n```sql\n{generated_sql}\n```"
        
        history[-1][1] = response_text
        return history, "", "✅ Processing completed!", ui_state.chart_figure
    
    elif ui_state.captured_output:
        # Show whatever output we have
        response_text = f"**Processing Steps:**\n```\n{ui_state.captured_output}\n```"
        history[-1][1] = response_text
        return history, "", "Processing in progress...", None
    
    return history, "", "Processing completed.", None

def check_for_updates(history):
    """Check for updates in processing"""
    if ui_state.waiting_for_input and ui_state.current_question and history:
        # Update the last message if it doesn't already show the question
        if len(history) > 0:
            current_response = history[-1][1]
            if "I need clarification:" not in current_response:
                response_text = f"**Processing Steps:**\n```\n{ui_state.captured_output}\n```\n\n"
                response_text += f"❓ **I need clarification:**\n\n{ui_state.current_question}\n\n"
                response_text += "👇 Please provide your response in the text box below and click 'Submit Response'."
                history[-1][1] = response_text
    
    elif ui_state.processing_complete and ui_state.final_result and history:
        # Update with final result if not already shown
        if len(history) > 0:
            current_response = history[-1][1]
            if "Final Answer:" not in current_response:
                response_text = f"**Processing Steps:**\n```\n{ui_state.captured_output}\n```\n\n"
                
                final_answer = ui_state.final_result.get('final_output', 'No result generated.')
                generated_sql = ui_state.final_result.get('validated_sql', '')
                
                response_text += f"**Final Answer:** {final_answer}"
                if generated_sql and generated_sql != 'No SQL generated.':
                    response_text += f"\n\n**Generated SQL:**\n```sql\n{generated_sql}\n```"
                
                history[-1][1] = response_text
    
    return history

# Create the Gradio interface
def create_app():
    with gr.Blocks(title="SQL Query Assistant with Smart Charts", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🔍 SQL Query Assistant with Smart Charts
        ### Ask questions about your database in natural language!
        
        I'll show you my thinking process, ask for clarification when needed, and create relevant visualizations automatically.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    height=600,
                    show_label=False,
                    container=True,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="Ask me anything about your database...",
                        label="Your Question",
                        scale=4,
                        lines=2
                    )
                    ask_btn = gr.Button("🚀 Ask", variant="primary", scale=1)
                
                with gr.Row():
                    response_input = gr.Textbox(
                        placeholder="When I ask for clarification, type your response here...",
                        label="Response to Clarification",
                        scale=4,
                        lines=2
                    )
                    respond_btn = gr.Button("📤 Submit Response", variant="secondary", scale=1)
                
                status_msg = gr.Textbox(label="Status", interactive=False, visible=True)
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="stop")
                    refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Smart Visualization")
                # chart_plot = gr.Plot(label="Auto-Generated Chart", visible=True)
                gr.Image('chart.png')
        
        # Sample queries section
        with gr.Accordion("💡 Sample Questions (Chart-Friendly)", open=True):
            gr.Markdown("These questions will generate charts automatically:")
            
            chart_samples = [
                "Total sales by distributor",
                "Monthly sales trend for bhujia", 
                "Top 10 products by revenue",
                "Sales comparison by region",
                "Product distribution breakdown",
                "Revenue trend over time"
            ]
            
            for i in range(0, len(chart_samples), 3):
                with gr.Row():
                    for j in range(3):
                        if i + j < len(chart_samples):
                            sample = chart_samples[i + j]
                            btn = gr.Button(sample, size="sm", scale=1)
                            btn.click(lambda s=sample: (s, ""), outputs=[query_input, status_msg])
        
        # Add a simple test chart button
        with gr.Accordion("🧪 Chart Testing", open=False):
            gr.Markdown("Test chart generation with sample data:")
            
            def create_test_chart():
                # Create sample data
                import pandas as pd
                test_data = {
                    'Product': ['Bhujia', 'Namkeen', 'Chips', 'Sweets'],
                    'Sales': [1500, 1200, 800, 600]
                }
                df = pd.DataFrame(test_data)
                
                fig = px.bar(df, x='Product', y='Sales', title="Test Chart - Sample Sales Data")
                fig.update_layout(height=400, template="plotly_white")
                
                return fig
            
            test_chart_btn = gr.Button("Generate Test Chart")
            # test_chart_btn.click(fn=create_test_chart, outputs=[chart_plot])
        
        # Chart rules info
        with gr.Accordion("📋 Chart Generation Rules", open=False):
            gr.Markdown("""
            **Charts are automatically created when:**
            - Query contains keywords like: sales, revenue, total, trend, comparison, by, top, distribution
            - Results have at least 2 rows and numeric data
            - Data is suitable for visualization
            
            **Chart Types:**
            - **Bar Charts**: Comparisons, rankings (e.g., "sales by distributor")
            - **Line Charts**: Trends over time (e.g., "monthly sales trend") 
            - **Pie Charts**: Distributions, breakdowns (e.g., "market share breakdown")
            """)
        
        # Event handlers
        ask_btn.click(
            fn=process_query,
            inputs=[query_input, chatbot],
            # outputs=[chatbot, query_input, chart_plot]
        )
        
        respond_btn.click(
            fn=process_user_response,
            inputs=[response_input, chatbot],
            # outputs=[chatbot, response_input, status_msg, chart_plot]
        )
        
        clear_btn.click(
            fn=lambda: ([], "", "", "", None),
            # outputs=[chatbot, query_input, response_input, status_msg, chart_plot]
        )
        
        refresh_btn.click(
            fn=check_for_updates,
            inputs=[chatbot],
            outputs=[chatbot]
        )
        
        query_input.submit(
            fn=process_query,
            inputs=[query_input, chatbot],
            # outputs=[chatbot, query_input, chart_plot]
        )
        
        response_input.submit(
            fn=process_user_response,
            inputs=[response_input, chatbot],
            # outputs=[chatbot, response_input, status_msg, chart_plot]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_app()
    demo.launch(
        server_port=7860,
        share=True,
        debug=True,
        show_error=True)