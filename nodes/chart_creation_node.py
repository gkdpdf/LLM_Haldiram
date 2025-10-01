from typing import TypedDict, Dict, Any, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from decimal import Decimal

def chart_creation_node(state):
    """
    Creates and saves chart as PNG in root folder
    """
    print("DEBUG: Chart creation node called!")
    
    user_query = state.get("user_query", "")
    execution_result = state.get("execution_result", [])
    execution_status = state.get("execution_status", "")
    
    # Skip chart creation if query failed or no results
    if execution_status != "success" or not execution_result:
        print("DEBUG: Skipping chart - no results or failed execution")
        return {"final_output": state.get("final_output", "")}
    
    try:
        # Convert results to DataFrame
        if isinstance(execution_result, list) and len(execution_result) > 0:
            df = pd.DataFrame(execution_result)
            print(f"DEBUG: DataFrame created with shape: {df.shape}")
            print(f"DEBUG: DataFrame columns: {df.columns.tolist()}")
            print(f"DEBUG: Data types before conversion: {df.dtypes}")
            print(f"DEBUG: Sample values before conversion:")
            for col in df.columns:
                sample_val = df[col].iloc[0]
                print(f"  {col}: {sample_val} (type: {type(sample_val)})")
            
            # Handle Decimal objects and convert only truly numeric columns
            for col in df.columns:
                # Check if column contains Decimal objects or numeric strings
                first_val = df[col].iloc[0]
                
                # Define numeric column patterns
                numeric_keywords = ['quantity', 'amount', 'sales', 'revenue', 'total', 'count', 
                                  'price', 'value', 'number', 'sum', 'avg', 'mean', 'rate', 'billed']
                
                # Check if column name suggests it's numeric
                is_numeric_column = any(keyword in col.lower() for keyword in numeric_keywords)
                
                # Check if the actual data is numeric (Decimal, int, float, or numeric string)
                is_numeric_data = isinstance(first_val, (Decimal, int, float)) or \
                                (isinstance(first_val, str) and first_val.replace(',', '').replace('.', '').replace('-', '').isdigit())
                
                if is_numeric_column or is_numeric_data:
                    try:
                        # Convert Decimal objects or numeric strings to proper numeric types
                        if isinstance(first_val, Decimal):
                            df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        else:
                            # Clean string data and convert
                            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce')
                        
                        print(f"DEBUG: Successfully converted {col} to numeric")
                    except Exception as conv_error:
                        print(f"DEBUG: Could not convert {col} to numeric: {conv_error}")
                else:
                    print(f"DEBUG: Keeping {col} as categorical (not numeric)")
            
            print(f"DEBUG: Data types after conversion: {df.dtypes}")
            print(f"DEBUG: Sample values after conversion:")
            for col in df.columns:
                sample_val = df[col].iloc[0]
                print(f"  {col}: {sample_val} (type: {type(sample_val)})")
                
        else:
            print("DEBUG: No valid execution result for DataFrame")
            return {"final_output": state.get("final_output", "")}
    except Exception as e:
        print(f"DEBUG: Error creating DataFrame: {e}")
        return {"final_output": state.get("final_output", "")}
    
    # Create chart
    try:
        # Get numeric and categorical columns after conversion
        numeric_cols = df.select_dtypes(include=['number', 'int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        print(f"DEBUG: Numeric columns after conversion: {numeric_cols}")
        print(f"DEBUG: Categorical columns after conversion: {categorical_cols}")
        print(f"DEBUG: DataFrame after conversions:")
        print(df.head())
        
        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            # Create bar chart
            x_col = categorical_cols[0]  # Use first categorical column for x-axis
            y_col = numeric_cols[0]      # Use first numeric column for y-axis
            
            print(f"DEBUG: Creating chart with x={x_col}, y={y_col}")
            
            # Remove any rows with NaN values in the columns we're using
            df_clean = df.dropna(subset=[x_col, y_col])
            print(f"DEBUG: Clean DataFrame shape: {df_clean.shape}")
            print(f"DEBUG: Clean DataFrame:")
            print(df_clean)
            
            if len(df_clean) == 0:
                print("DEBUG: No data left after cleaning NaN values")
                return {"final_output": state.get("final_output", "")}
            
            # Limit to top 10 if more than 10 records
            if len(df_clean) > 10:
                df_chart = df_clean.nlargest(10, y_col)
                title_suffix = " (Top 10)"
            else:
                df_chart = df_clean
                title_suffix = ""
            
            # Create the chart
            fig = px.bar(df_chart, 
                        x=x_col, 
                        y=y_col, 
                        title=f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}{title_suffix}")
            
            # Improve chart formatting
            fig.update_xaxes(tickangle=45, title_font_size=12)
            fig.update_yaxes(title_font_size=12)
            fig.update_layout(
                margin=dict(b=150, l=60, r=60, t=60),  # Add margins
                font=dict(size=10),
                showlegend=False
            )
            
            print("DEBUG: Chart figure created, attempting to save...")
            
            # Save as PNG in root folder
            fig.write_image("chart.png", width=1000, height=700)
            
            print("Chart saved successfully as chart.png in root folder")
            
            # Update final output
            current_output = state.get("final_output", "")
            chart_info = f"\n\n📊 Chart created successfully! Saved as chart.png showing {y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}"
            
            return {"final_output": current_output + chart_info}
            
        else:
            print(f"DEBUG: Cannot create chart - numeric_cols: {len(numeric_cols)}, categorical_cols: {len(categorical_cols)}")
            
            # If we still have issues, let's try a different approach
            if len(df.columns) >= 2:
                # Manually assign columns based on their names and content
                text_col = None
                num_col = None
                
                for col in df.columns:
                    if 'name' in col.lower() or 'title' in col.lower():
                        text_col = col
                    elif any(keyword in col.lower() for keyword in ['quantity', 'amount', 'sales', 'total']):
                        num_col = col
                
                if text_col and num_col:
                    print(f"DEBUG: Manual assignment - text: {text_col}, numeric: {num_col}")
                    
                    # Ensure the numeric column is actually numeric
                    try:
                        df[num_col] = pd.to_numeric(df[num_col], errors='coerce')
                        df_clean = df.dropna(subset=[text_col, num_col])
                        
                        if len(df_clean) > 0:
                            fig = px.bar(df_clean, 
                                        x=text_col, 
                                        y=num_col, 
                                        title=f"{num_col.replace('_', ' ').title()} by {text_col.replace('_', ' ').title()}")
                            
                            fig.update_xaxes(tickangle=45)
                            fig.update_layout(margin=dict(b=150))
                            fig.write_image("chart.png", width=1000, height=700)
                            
                            current_output = state.get("final_output", "")
                            chart_info = f"\n\n📊 Chart created successfully! Saved as chart.png"
                            return {"final_output": current_output + chart_info}
                    except Exception as manual_error:
                        print(f"DEBUG: Manual chart creation failed: {manual_error}")
    
    except Exception as e:
        print(f"Error creating chart: {str(e)}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
    
    # Return original output if no chart created
    print("DEBUG: No chart could be created")
    return {"final_output": state.get("final_output", "")}
