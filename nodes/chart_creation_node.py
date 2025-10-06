from typing import TypedDict, Dict, Any, List, Optional, Literal
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from decimal import Decimal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import psycopg2

# Chart recommendation schema
class ChartRecommendation(BaseModel):
    """Structured output for chart type recommendation"""
    chart_type: Literal["bar", "line", "pie", "scatter", "area", "comparison_bar", "trend_line"] = Field(
        description="The most appropriate chart type"
    )
    x_column: str = Field(description="Column for x-axis")
    y_column: str = Field(description="Column for y-axis")
    title: str = Field(description="Chart title")
    reasoning: str = Field(description="Why this chart type")
    should_add_context: bool = Field(
        default=False,
        description="Whether to add comparative context (other products/distributors)"
    )
    context_type: Optional[Literal["top_products", "other_distributors", "trend_over_time"]] = Field(
        default=None,
        description="Type of comparative context to add"
    )
    time_period: Optional[str] = Field(
        default=None,
        description="Time period if temporal analysis (e.g., '2 months', 'weekly')"
    )


def enhance_data_with_context(df: pd.DataFrame, state: dict, recommendation: ChartRecommendation) -> pd.DataFrame:
    """
    Enhance single-row results with comparative context
    """
    resolved = state.get("resolved", {})
    entities = resolved.get("entities", {})
    filters = resolved.get("filters", {})
    table = resolved.get("table")
    
    # Get database connection
    conn = psycopg2.connect(
        host="localhost",
        dbname="haldiram",
        user="postgres",
        password="12345678"
    )
    
    try:
        # CASE 1: Single product sales - add top products for comparison
        if recommendation.should_add_context and recommendation.context_type == "top_products":
            product_entity = entities.get("product", {})
            product_name = product_entity.get("value") if isinstance(product_entity, dict) else None
            
            if product_name and table == "tbl_primary":
                print(f"📊 Adding context: Top 5 products alongside {product_name}")
                
                # Build query for top products
                time_filter = ""
                if filters.get("time_range"):
                    time_filter = f"AND sales_order_date BETWEEN '{filters['time_range'][0]}' AND '{filters['time_range'][1]}'"
                
                query = f"""
                SELECT product_name, SUM(invoiced_total_quantity) as total_sales
                FROM tbl_primary
                WHERE 1=1 {time_filter}
                GROUP BY product_name
                ORDER BY total_sales DESC
                LIMIT 6
                """
                
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
                cursor.close()
                
                enhanced_df = pd.DataFrame(results, columns=['product_name', 'total_sales'])
                
                # Highlight the queried product
                enhanced_df['highlight'] = enhanced_df['product_name'] == product_name
                return enhanced_df
        
        # CASE 2: Single distributor sales - add other distributors
        elif recommendation.should_add_context and recommendation.context_type == "other_distributors":
            distributor_entity = entities.get("distributor", {})
            distributor_name = distributor_entity.get("value") if isinstance(distributor_entity, dict) else None
            
            if distributor_name and table == "tbl_primary":
                print(f"📊 Adding context: Top 5 distributors alongside {distributor_name}")
                
                time_filter = ""
                if filters.get("time_range"):
                    time_filter = f"AND sales_order_date BETWEEN '{filters['time_range'][0]}' AND '{filters['time_range'][1]}'"
                
                query = f"""
                SELECT distributor_name, SUM(invoiced_total_quantity) as total_sales
                FROM tbl_primary
                WHERE 1=1 {time_filter}
                GROUP BY distributor_name
                ORDER BY total_sales DESC
                LIMIT 6
                """
                
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
                cursor.close()
                
                enhanced_df = pd.DataFrame(results, columns=['distributor_name', 'total_sales'])
                enhanced_df['highlight'] = enhanced_df['distributor_name'] == distributor_name
                return enhanced_df
        
        # CASE 3: Time-based query - show trend
        elif recommendation.should_add_context and recommendation.context_type == "trend_over_time":
            product_entity = entities.get("product", {})
            product_name = product_entity.get("value") if isinstance(product_entity, dict) else None
            
            if product_name and table == "tbl_primary" and filters.get("time_range"):
                print(f"📊 Adding context: Trend analysis for {product_name}")
                
                # Weekly trend over the time period
                query = f"""
                SELECT 
                    DATE_TRUNC('week', sales_order_date) as week,
                    SUM(invoiced_total_quantity) as total_sales
                FROM tbl_primary
                WHERE product_name = '{product_name}'
                    AND sales_order_date BETWEEN '{filters['time_range'][0]}' AND '{filters['time_range'][1]}'
                GROUP BY DATE_TRUNC('week', sales_order_date)
                ORDER BY week
                """
                
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
                cursor.close()
                
                if results:
                    enhanced_df = pd.DataFrame(results, columns=['week', 'total_sales'])
                    enhanced_df['week'] = pd.to_datetime(enhanced_df['week'])
                    return enhanced_df
        
    except Exception as e:
        print(f"⚠️ Error enhancing data: {e}")
    finally:
        conn.close()
    
    return df


def get_smart_chart_recommendation(user_query: str, df: pd.DataFrame, state: dict) -> Optional[ChartRecommendation]:
    """
    Enhanced LLM recommendation with context awareness
    """
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(ChartRecommendation)
        
        # Extract query context
        resolved = state.get("resolved", {})
        entities = resolved.get("entities", {})
        filters = resolved.get("filters", {})
        
        has_product = "product" in entities
        has_distributor = "distributor" in entities
        has_time_filter = "time_range" in filters
        single_row = len(df) == 1
        
        data_summary = {
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "is_single_result": single_row,
            "has_product_filter": has_product,
            "has_distributor_filter": has_distributor,
            "has_time_filter": has_time_filter,
            "sample_data": df.head(3).to_dict('records')
        }
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data visualization expert. Recommend charts with comparative context.

Guidelines:
1. **Single product sales** → Set should_add_context=True, context_type="top_products" 
   - Show this product alongside top 5 products for comparison
   
2. **Single distributor sales** → Set should_add_context=True, context_type="other_distributors"
   - Show this distributor alongside top 5 distributors
   
3. **Product sales with time filter (month/weeks)** → Set should_add_context=True, context_type="trend_over_time"
   - Show weekly/daily trend line instead of single bar
   
4. **Multiple results** → Use regular bar/line charts, no context needed

Chart types:
- comparison_bar: For single items with comparative context
- trend_line: For time-based analysis
- bar: For regular comparisons
- line: For trends without context enhancement"""),
            ("user", """User Query: {query}

Data Context:
- Rows: {row_count}
- Single result: {is_single}
- Product filter: {has_product}
- Distributor filter: {has_distributor}
- Time filter: {has_time}

Sample Data: {sample_data}

Recommend the best chart with appropriate context.""")
        ])
        
        formatted_prompt = prompt.format_messages(
            query=user_query,
            row_count=data_summary['row_count'],
            is_single=data_summary['is_single_result'],
            has_product=data_summary['has_product_filter'],
            has_distributor=data_summary['has_distributor_filter'],
            has_time=data_summary['has_time_filter'],
            sample_data=str(data_summary['sample_data'])
        )
        
        recommendation = structured_llm.invoke(formatted_prompt)
        
        print(f"Chart Recommendation: {recommendation.chart_type}")
        print(f"Add Context: {recommendation.should_add_context} ({recommendation.context_type})")
        print(f"Reasoning: {recommendation.reasoning}")
        
        return recommendation
        
    except Exception as e:
        print(f"Error getting recommendation: {e}")
        return None


def create_enhanced_chart(df: pd.DataFrame, recommendation: ChartRecommendation, state: dict, output_path: str = "chart.png"):
    """
    Create chart with optional comparative context
    """
    try:
        # Enhance data if context requested
        if recommendation.should_add_context:
            df = enhance_data_with_context(df, state, recommendation)
            
            if df is None or len(df) == 0:
                print("Could not enhance data, using original")
                df = state.get("execution_result", [])
                df = pd.DataFrame(df)
        
        # Clean data
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].dtype == object:
                try:
                    df_clean[col] = df_clean[col].apply(
                        lambda x: float(x) if isinstance(x, Decimal) else x
                    )
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
                except:
                    pass
        
        # Create chart
        fig = None
        
        if recommendation.chart_type in ["bar", "comparison_bar"]:
            # Check if we have highlight column
            if 'highlight' in df_clean.columns:
                colors = ['#FF4444' if h else '#4A90E2' for h in df_clean['highlight']]
                fig = go.Figure(data=[
                    go.Bar(
                        x=df_clean[recommendation.x_column],
                        y=df_clean[recommendation.y_column],
                        marker_color=colors,
                        text=df_clean[recommendation.y_column],
                        textposition='outside'
                    )
                ])
                fig.update_layout(title=recommendation.title)
            else:
                fig = px.bar(df_clean, x=recommendation.x_column, y=recommendation.y_column,
                           title=recommendation.title)
            
            fig.update_xaxes(tickangle=45)
            
        elif recommendation.chart_type in ["line", "trend_line"]:
            fig = px.line(df_clean, x=recommendation.x_column, y=recommendation.y_column,
                         title=recommendation.title, markers=True)
        
        if fig is None:
            return False
        
        fig.update_layout(
            margin=dict(b=150, l=80, r=80, t=80),
            font=dict(size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        
        fig.write_image(output_path, width=1200, height=700)
        print(f"Chart saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating chart: {e}")
        import traceback
        traceback.print_exc()
        return False


def chart_creation_node(state):
    """
    Enhanced chart node with smart context-aware visualizations
    """
    print("Chart creation node started")
    
    execution_result = state.get("execution_result", [])
    execution_status = state.get("execution_status", "")
    
    if execution_status != "success" or not execution_result:
        print("Skipping chart - no results")
        return {"final_output": state.get("final_output", "")}
    
    try:
        df = pd.DataFrame(execution_result)
        print(f"DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Get smart recommendation
        recommendation = get_smart_chart_recommendation(
            state.get("user_query", ""), 
            df, 
            state
        )
        
        if recommendation:
            success = create_enhanced_chart(df, recommendation, state)
            
            if success:
                current_output = state.get("final_output", "")
                chart_info = f"\n\n📊 **Chart Created**: {recommendation.title}\n"
                chart_info += f"Type: {recommendation.chart_type.replace('_', ' ').title()}\n"
                chart_info += f"File: chart.png\n"
                if recommendation.should_add_context:
                    chart_info += f"Context: Showing comparative {recommendation.context_type.replace('_', ' ')}\n"
                chart_info += f"Insight: {recommendation.reasoning}"
                
                return {"final_output": current_output + chart_info}
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    return {"final_output": state.get("final_output", "")}