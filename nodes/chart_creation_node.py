from typing import TypedDict, Dict, Any, List, Optional, Literal
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from decimal import Decimal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Define structured output schema for chart recommendations
class ChartRecommendation(BaseModel):
    """Structured output for chart type recommendation"""
    chart_type: Literal["bar", "line", "pie", "scatter", "heatmap", "box", "histogram", "area"] = Field(
        description="The most appropriate chart type for the data and query"
    )
    x_column: str = Field(description="Column name to use for x-axis")
    y_column: str = Field(description="Column name to use for y-axis (or values for pie chart)")
    title: str = Field(description="Descriptive title for the chart")
    reasoning: str = Field(description="Brief explanation of why this chart type was chosen")
    color_column: Optional[str] = Field(default=None, description="Optional column for color grouping")
    aggregate_function: Optional[Literal["sum", "avg", "count", "max", "min"]] = Field(
        default=None, 
        description="Aggregation function if data needs to be grouped"
    )

def get_chart_recommendation(user_query: str, df: pd.DataFrame, llm_model: str = "gpt-4") -> Optional[ChartRecommendation]:
    """
    Use LLM with structured output to recommend the best chart type
    """
    try:
        # Initialize LLM with structured output
        llm = ChatOpenAI(model=llm_model, temperature=0)
        structured_llm = llm.with_structured_output(ChartRecommendation)
        
        # Prepare data summary
        data_summary = {
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "shape": df.shape,
            "sample_data": df.head(3).to_dict('records'),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'string']).columns.tolist()
        }
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert data visualization assistant. Analyze the user's query and data to recommend the best chart type.

Chart type guidelines:
- **bar**: Compare categories, show rankings, compare discrete values
- **line**: Show trends over time, continuous data progression
- **pie**: Show composition/proportions of a whole (use sparingly, max 5-7 slices)
- **scatter**: Show relationships between two continuous variables
- **heatmap**: Show patterns in matrix data or correlations
- **box**: Show distribution and outliers
- **histogram**: Show frequency distribution of continuous data
- **area**: Show cumulative trends over time

Consider:
1. The user's intent from their query
2. Data types (temporal, categorical, continuous)
3. Number of data points
4. Relationships being explored"""),
            ("user", """User Query: {query}

Data Summary:
- Columns: {columns}
- Data Types: {dtypes}
- Rows: {rows}
- Numeric Columns: {numeric_cols}
- Categorical Columns: {categorical_cols}

Sample Data (first 3 rows):
{sample_data}

Recommend the best chart type and configuration.""")
        ])
        
        # Format and invoke
        formatted_prompt = prompt.format_messages(
            query=user_query,
            columns=data_summary['columns'],
            dtypes=data_summary['dtypes'],
            rows=data_summary['shape'][0],
            numeric_cols=data_summary['numeric_columns'],
            categorical_cols=data_summary['categorical_columns'],
            sample_data=str(data_summary['sample_data'])
        )
        
        recommendation = structured_llm.invoke(formatted_prompt)
        
        print(f"Chart Recommendation: {recommendation.chart_type}")
        print(f"Reasoning: {recommendation.reasoning}")
        
        return recommendation
        
    except Exception as e:
        print(f"Error getting chart recommendation: {e}")
        return None

def create_chart_from_recommendation(df: pd.DataFrame, recommendation: ChartRecommendation, output_path: str = "chart.png"):
    """
    Create a chart based on LLM recommendation
    """
    try:
        # Clean data
        df_clean = df.copy()
        
        # Handle Decimal objects
        for col in df_clean.columns:
            if df_clean[col].dtype == object:
                try:
                    # Try to convert Decimal or numeric strings
                    df_clean[col] = df_clean[col].apply(
                        lambda x: float(x) if isinstance(x, Decimal) else x
                    )
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
                except:
                    pass
        
        # Apply aggregation if specified
        if recommendation.aggregate_function and recommendation.color_column:
            agg_dict = {recommendation.y_column: recommendation.aggregate_function}
            df_clean = df_clean.groupby([recommendation.x_column, recommendation.color_column]).agg(agg_dict).reset_index()
        elif recommendation.aggregate_function:
            agg_dict = {recommendation.y_column: recommendation.aggregate_function}
            df_clean = df_clean.groupby(recommendation.x_column).agg(agg_dict).reset_index()
        
        # Remove NaN values
        required_cols = [recommendation.x_column, recommendation.y_column]
        if recommendation.color_column:
            required_cols.append(recommendation.color_column)
        df_clean = df_clean.dropna(subset=required_cols)
        
        if len(df_clean) == 0:
            print("No data after cleaning")
            return False
        
        # Limit data points for readability
        max_points = 50
        if len(df_clean) > max_points and recommendation.chart_type in ['bar', 'pie']:
            df_clean = df_clean.nlargest(max_points, recommendation.y_column)
        
        # Create chart based on type
        fig = None
        
        if recommendation.chart_type == "bar":
            fig = px.bar(
                df_clean,
                x=recommendation.x_column,
                y=recommendation.y_column,
                color=recommendation.color_column,
                title=recommendation.title
            )
            fig.update_xaxes(tickangle=45)
            
        elif recommendation.chart_type == "line":
            fig = px.line(
                df_clean,
                x=recommendation.x_column,
                y=recommendation.y_column,
                color=recommendation.color_column,
                title=recommendation.title,
                markers=True
            )
            
        elif recommendation.chart_type == "pie":
            fig = px.pie(
                df_clean,
                names=recommendation.x_column,
                values=recommendation.y_column,
                title=recommendation.title
            )
            
        elif recommendation.chart_type == "scatter":
            fig = px.scatter(
                df_clean,
                x=recommendation.x_column,
                y=recommendation.y_column,
                color=recommendation.color_column,
                title=recommendation.title,
                size=recommendation.y_column if len(df_clean) < 100 else None
            )
            
        elif recommendation.chart_type == "area":
            fig = px.area(
                df_clean,
                x=recommendation.x_column,
                y=recommendation.y_column,
                color=recommendation.color_column,
                title=recommendation.title
            )
            
        elif recommendation.chart_type == "histogram":
            fig = px.histogram(
                df_clean,
                x=recommendation.x_column,
                title=recommendation.title,
                nbins=30
            )
            
        elif recommendation.chart_type == "box":
            fig = px.box(
                df_clean,
                x=recommendation.x_column,
                y=recommendation.y_column,
                color=recommendation.color_column,
                title=recommendation.title
            )
        
        if fig is None:
            print(f"Chart type {recommendation.chart_type} not implemented")
            return False
        
        # Improve formatting
        fig.update_layout(
            margin=dict(b=150, l=80, r=80, t=80),
            font=dict(size=11),
            showlegend=True if recommendation.color_column else False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Save chart
        fig.write_image(output_path, width=1200, height=700)
        print(f"Chart saved successfully: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error creating chart: {e}")
        import traceback
        traceback.print_exc()
        return False

def chart_creation_node(state):
    """
    Enhanced chart creation node using LLM for intelligent chart selection
    """
    print("Chart creation node started")
    
    user_query = state.get("user_query", "")
    execution_result = state.get("execution_result", [])
    execution_status = state.get("execution_status", "")
    
    # Skip if no valid results
    if execution_status != "success" or not execution_result:
        print("Skipping chart - no results or failed execution")
        return {"final_output": state.get("final_output", "")}
    
    try:
        # Convert to DataFrame
        if isinstance(execution_result, list) and len(execution_result) > 0:
            df = pd.DataFrame(execution_result)
            print(f"DataFrame created: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Handle Decimal conversions
            for col in df.columns:
                if df[col].dtype == object:
                    first_val = df[col].iloc[0] if len(df) > 0 else None
                    if isinstance(first_val, Decimal):
                        df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
                        df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Get LLM recommendation
            recommendation = get_chart_recommendation(user_query, df)
            
            if recommendation:
                # Create chart based on recommendation
                success = create_chart_from_recommendation(df, recommendation)
                
                if success:
                    current_output = state.get("final_output", "")
                    chart_info = f"\n\n📊 **Chart Created**: {recommendation.title}\n"
                    chart_info += f"Type: {recommendation.chart_type.title()} Chart\n"
                    chart_info += f"File: chart.png\n"
                    chart_info += f"Insight: {recommendation.reasoning}"
                    
                    return {"final_output": current_output + chart_info}
            else:
                print("Could not get chart recommendation, using fallback")
                # Fallback to original logic if LLM fails
                return fallback_chart_creation(df, state)
        
    except Exception as e:
        print(f"Error in chart creation: {e}")
        import traceback
        traceback.print_exc()
    
    return {"final_output": state.get("final_output", "")}

def fallback_chart_creation(df: pd.DataFrame, state: dict):
    """
    Fallback chart creation if LLM recommendation fails
    """
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            x_col = categorical_cols[0]
            y_col = numeric_cols[0]
            
            df_clean = df.dropna(subset=[x_col, y_col])
            if len(df_clean) > 10:
                df_clean = df_clean.nlargest(10, y_col)
            
            fig = px.bar(df_clean, x=x_col, y=y_col, 
                        title=f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}")
            fig.update_xaxes(tickangle=45)
            fig.update_layout(margin=dict(b=150))
            fig.write_image("chart.png", width=1000, height=700)
            
            current_output = state.get("final_output", "")
            chart_info = "\n\n📊 Chart created successfully! Saved as chart.png"
            return {"final_output": current_output + chart_info}
            
    except Exception as e:
        print(f"Fallback chart creation failed: {e}")
    
    return {"final_output": state.get("final_output", "")}