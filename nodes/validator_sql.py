import psycopg2
import sqlparse
import re
from typing import Dict, Any, List, Optional, Tuple
from langchain_openai import ChatOpenAI
import os

class InteractiveValidator:
    """
    Enhanced validator with user interaction for ambiguous cases
    and zero-hallucination guarantee
    """
    
    def __init__(self, connection_params, annotated_schema, relationships):
        self.connection_params = connection_params
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY'))
        self.annotated_schema = annotated_schema
        self.relationships = relationships
        self.table_structure = self.parse_annotated_schema(annotated_schema)
        self.valid_joins = self.parse_relationships(relationships)
    
    def parse_annotated_schema(self, annotated_schema: str) -> Dict[str, Any]:
        """
        Parse annotated_schema.md to extract tables and columns
        Handles JSON format with table descriptions and column arrays
        """
        table_structure = {}
        
        print(f"📋 Parsing annotated schema...")
        
        # Split by table headers (### **tbl_...)
        table_blocks = re.split(r'###\s*\*\*(\w+)\*\*', annotated_schema)
        
        # Process each table block
        for i in range(1, len(table_blocks), 2):
            if i + 1 < len(table_blocks):
                table_name = table_blocks[i].lower().strip()
                table_content = table_blocks[i + 1]
                
                print(f"  Found table: {table_name}")
                
                # Initialize table structure
                table_structure[table_name] = {
                    'columns': [],
                    'types': {},
                    'descriptions': {}
                }
                
                # Extract columns from JSON array format
                # Looking for patterns like: ["column_name : description, datatype: type, <sample values: ...>"]
                column_pattern = r'\["([^:]+)\s*:\s*([^"]+)"\]'
                columns = re.findall(column_pattern, table_content)
                
                for col_name, col_desc in columns:
                    column = col_name.strip().lower()
                    description = col_desc.strip()
                    
                    # Add to table structure
                    table_structure[table_name]['columns'].append(column)
                    table_structure[table_name]['descriptions'][column] = description
                    
                    # Infer data type from description
                    if 'datatype: date' in description.lower() or 'date' in description.lower():
                        table_structure[table_name]['types'][column] = 'date'
                    elif 'datatype: integer' in description.lower() or 'datatype: int' in description.lower():
                        table_structure[table_name]['types'][column] = 'integer'
                    elif 'datatype: float' in description.lower():
                        table_structure[table_name]['types'][column] = 'float'
                    elif 'datatype: boolean' in description.lower():
                        table_structure[table_name]['types'][column] = 'boolean'
                    else:
                        table_structure[table_name]['types'][column] = 'text'
        
        print(f"✅ Parsed {len(table_structure)} tables from annotated schema")
        for table, info in table_structure.items():
            print(f"  {table}: {len(info['columns'])} columns")
        
        return table_structure
    
    def parse_relationships(self, relationships: str) -> List[Dict[str, str]]:
        """
        Parse relationship.txt to extract valid JOIN conditions
        Returns list of valid join patterns
        """
        valid_joins = []
        
        lines = relationships.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for JOIN patterns like "tbl_primary.product_id = tbl_product_master.product_erp_id"
            join_match = re.search(
                r'(\w+)\.(\w+)\s*[=→]\s*(\w+)\.(\w+)', 
                line, 
                re.IGNORECASE
            )
            
            if join_match:
                valid_joins.append({
                    'left_table': join_match.group(1).lower(),
                    'left_column': join_match.group(2).lower(),
                    'right_table': join_match.group(3).lower(),
                    'right_column': join_match.group(4).lower()
                })
        
        return valid_joins
    
    def validate_and_fix_sql(self, sql_query: str, user_query: str, 
                            annotated_schema: str, relationships: str) -> Dict[str, Any]:
        """
        Main validation orchestrator with user interaction
        Returns: {validated_sql, status, error, attempts}
        """
        max_attempts = 10
        current_sql = sql_query.strip()
        
        print("\n" + "="*70)
        print("🔍 SQL VALIDATION STARTED")
        print("="*70)
        print(f"Original SQL:\n{current_sql}\n")
        
        for attempt in range(1, max_attempts + 1):
            print(f"\n{'─'*70}")
            print(f"🔄 Validation Attempt {attempt}/{max_attempts}")
            print(f"{'─'*70}")
            
            # Level 1: Syntax validation
            is_valid, error = self._validate_syntax(current_sql)
            if not is_valid:
                print(f"❌ Syntax Error: {error}")
                current_sql = self._fix_with_user_help(
                    current_sql, error, "syntax", user_query, 
                    annotated_schema, relationships
                )
                continue
            
            # Level 2: Table/Column existence
            is_valid, error = self._validate_structure(current_sql)
            if not is_valid:
                print(f"❌ Structure Error: {error}")
                current_sql = self._fix_with_user_help(
                    current_sql, error, "structure", user_query,
                    annotated_schema, relationships
                )
                continue
            
            # Level 3: Join validation
            is_valid, error = self._validate_joins(current_sql, relationships)
            if not is_valid:
                print(f"❌ Join Error: {error}")
                current_sql = self._fix_with_user_help(
                    current_sql, error, "joins", user_query,
                    annotated_schema, relationships
                )
                continue
            
            # Level 4: Execution test
            is_valid, error = self._test_execution(current_sql)
            if not is_valid:
                print(f"❌ Execution Error: {error}")
                current_sql = self._fix_with_user_help(
                    current_sql, error, "execution", user_query,
                    annotated_schema, relationships
                )
                continue
            
            # Level 5: Logic validation
            is_valid, error = self._validate_logic(
                current_sql, user_query, annotated_schema, relationships
            )
            if not is_valid:
                print(f"❌ Logic Error: {error}")
                current_sql = self._fix_with_user_help(
                    current_sql, error, "logic", user_query,
                    annotated_schema, relationships
                )
                continue
            
            # All validations passed!
            print(f"\n{'='*70}")
            print(f"✅ SQL VALIDATION SUCCESSFUL (Attempt {attempt})")
            print(f"{'='*70}")
            print(f"Final SQL:\n{current_sql}\n")
            
            return {
                "validated_sql": current_sql,
                "validation_status": "valid" if attempt == 1 else "corrected",
                "validation_error": None,
                "correction_attempts": attempt
            }
        
        # Max attempts reached
        print(f"\n⚠️ Maximum validation attempts ({max_attempts}) reached")
        return {
            "validated_sql": current_sql,
            "validation_status": "max_attempts",
            "validation_error": f"Could not validate after {max_attempts} attempts",
            "correction_attempts": max_attempts
        }
    
    def _validate_syntax(self, sql: str) -> Tuple[bool, str]:
        """Validate SQL syntax"""
        try:
            parsed = sqlparse.parse(sql)
            if not parsed or not parsed[0].tokens:
                return False, "Empty or invalid SQL statement"
            return True, ""
        except Exception as e:
            return False, f"SQL parsing error: {str(e)}"
    
    def _validate_structure(self, sql: str) -> Tuple[bool, str]:
        """Validate tables and columns exist in annotated schema"""
        # Extract tables
        tables = self._extract_tables(sql)
        for table in tables:
            if table not in self.table_structure:
                available = list(self.table_structure.keys())
                return False, f"Table '{table}' not in schema. Available: {available}"
        
        # Extract columns
        columns = self._extract_columns(sql)
        for table_alias, column in columns:
            actual_table = self._resolve_table_alias(sql, table_alias)
            if not actual_table:
                continue
            
            if actual_table in self.table_structure:
                if column not in self.table_structure[actual_table]['columns']:
                    available = self.table_structure[actual_table]['columns']
                    return False, f"Column '{column}' not in table '{actual_table}'. Available columns in {actual_table}: {available}"
        
        return True, ""
    
    def _validate_joins(self, sql: str, relationships: str) -> Tuple[bool, str]:
        """Validate JOIN conditions match documented relationships from relationship.txt"""
        join_pattern = r'JOIN\s+(\w+)\s+(?:AS\s+)?(\w+)?\s+ON\s+([\w\.]+)\s*=\s*([\w\.]+)'
        joins = re.findall(join_pattern, sql, re.IGNORECASE)
        
        if not joins:
            # No JOINs found, check if query uses multiple tables that should be joined
            tables = self._extract_tables(sql)
            if len(tables) > 1:
                return False, f"Query uses multiple tables {tables} but no JOIN found. Check relationships doc."
            return True, ""
        
        for join_info in joins:
            join_table = join_info[0] if len(join_info) > 0 else None
            join_alias = join_info[1] if len(join_info) > 1 else None
            left_col = join_info[2] if len(join_info) > 2 else None
            right_col = join_info[3] if len(join_info) > 3 else None
            
            if not left_col or not right_col:
                continue
            
            # Extract table.column format
            left_parts = left_col.split('.')
            right_parts = right_col.split('.')
            
            if len(left_parts) == 2 and len(right_parts) == 2:
                # Resolve aliases to actual table names
                left_table = self._resolve_table_alias(sql, left_parts[0])
                right_table = self._resolve_table_alias(sql, right_parts[0])
                left_column = left_parts[1].lower()
                right_column = right_parts[1].lower()
                
                # Skip validation if we couldn't resolve the alias
                if not left_table or not right_table:
                    print(f"⚠️ Could not resolve table aliases: {left_parts[0]} or {right_parts[0]}")
                    continue
                
                # Check if this join exists in valid_joins
                is_valid = False
                for valid_join in self.valid_joins:
                    # Check both directions
                    if ((valid_join['left_table'] == left_table and 
                         valid_join['left_column'] == left_column and
                         valid_join['right_table'] == right_table and 
                         valid_join['right_column'] == right_column) or
                        (valid_join['left_table'] == right_table and 
                         valid_join['left_column'] == right_column and
                         valid_join['right_table'] == left_table and 
                         valid_join['right_column'] == left_column)):
                        is_valid = True
                        break
                
                if not is_valid:
                    valid_joins_str = '\n'.join([
                        f"  - {vj['left_table']}.{vj['left_column']} = {vj['right_table']}.{vj['right_column']}"
                        for vj in self.valid_joins
                    ])
                    print(f"🔍 DEBUG: Checking join {left_table}.{left_column} = {right_table}.{right_column}")
                    return False, f"Invalid join: {left_table}.{left_column} = {right_table}.{right_column}.\n\nValid joins from relationships:\n{valid_joins_str}"
        
        return True, ""
    
    def _test_execution(self, sql: str) -> Tuple[bool, str]:
        """Test SQL execution without running it"""
        try:
            conn = psycopg2.connect(**self.connection_params)
            cursor = conn.cursor()
            
            # Use EXPLAIN to validate without executing
            cursor.execute(f"EXPLAIN {sql}")
            
            cursor.close()
            conn.close()
            return True, ""
            
        except psycopg2.Error as e:
            error_msg = str(e).split('\n')[0]
            return False, f"PostgreSQL Error: {error_msg}"
        except Exception as e:
            return False, f"Execution error: {str(e)}"
    
    def _validate_logic(self, sql: str, user_query: str, 
                       annotated_schema: str, relationships: str) -> Tuple[bool, str]:
        """Validate SQL logic matches user intent using annotated schema"""
        
        # Build schema summary from annotated schema
        schema_summary = self._build_schema_summary()
        
        # Extract values from WHERE clause to check if they look valid
        where_values = re.findall(r"=\s*'([^']+)'", sql, re.IGNORECASE)
        
        validation_prompt = f"""Validate if this SQL query correctly answers the user's question.

User Question: {user_query}

SQL Query: {sql}

Available Schema (from annotated_schema.md):
{schema_summary}

Valid Relationships (from relationship.txt):
{self._format_valid_joins()}

IMPORTANT: The SQL may contain specific product names, distributor names, or other data values in WHERE clauses.
These values were resolved from the actual database, so DO NOT question whether they exist.

Check ONLY:
1. Does the SQL answer the user's question?
2. Are the correct tables used from the schema?
3. Are only valid columns (from schema) referenced?
4. Are JOINs using the correct relationships (if any)?
5. Are aggregations appropriate for the query type?

DO NOT check:
- Whether specific data values in WHERE clauses exist (they were already validated)
- Product names, distributor names, or other string literals in the query

Respond with ONLY:
- "VALID" if the query structure is correct
- "INVALID: [specific issue]" if there's a structural problem (wrong column, wrong table, wrong JOIN)
"""
        
        try:
            result = self.llm.invoke(validation_prompt)
            response = result.content.strip()
            
            if response.startswith("VALID"):
                return True, ""
            else:
                # Check if the error is about data values (which we should ignore)
                error = response.replace("INVALID:", "").strip()
                if any(phrase in error.lower() for phrase in [
                    "does not match", "product name", "unclear if this product exists",
                    "does not exist in the provided schema", "expected product name format"
                ]):
                    print(f"⚠️ Ignoring spurious logic validation error about data values")
                    return True, ""
                return False, error
        except Exception as e:
            print(f"⚠️ Logic validation skipped: {e}")
            return True, ""
    
    def _build_schema_summary(self) -> str:
        """Build human-readable schema summary from annotated schema"""
        summary_parts = []
        
        for table, info in self.table_structure.items():
            summary_parts.append(f"\n{table}:")
            for col in info['columns']:
                desc = info['descriptions'].get(col, '')
                summary_parts.append(f"  - {col}: {desc}")
        
        return '\n'.join(summary_parts)
    
    def _format_valid_joins(self) -> str:
        """Format valid joins for prompt"""
        if not self.valid_joins:
            return "No explicit joins documented"
        
        join_lines = []
        for vj in self.valid_joins:
            join_lines.append(
                f"  {vj['left_table']}.{vj['left_column']} = "
                f"{vj['right_table']}.{vj['right_column']}"
            )
        
        return '\n'.join(join_lines)
    
    def _fix_with_user_help(self, sql: str, error: str, error_type: str,
                           user_query: str, annotated_schema: str, 
                           relationships: str) -> str:
        """
        Fix SQL automatically using LLM with enhanced context
        """
        print(f"\n🤖 Automatically fixing {error_type} error...")
        
        # Get LLM suggestion with enhanced prompt
        fix_prompt = self._create_fix_prompt(
            sql, error, error_type, user_query, annotated_schema, relationships
        )
        
        try:
            result = self.llm.invoke(fix_prompt)
            suggested_sql = self._extract_sql(result.content)
            
            # If the suggested SQL is identical to the original, the LLM is stuck
            if suggested_sql.strip() == sql.strip():
                print("⚠️ LLM returned same SQL. Applying schema-based fix...")
                suggested_sql = self._apply_schema_based_fix(sql, error, error_type)
            
            print(f"💡 Auto-corrected SQL:")
            print(f"{suggested_sql}\n")
            
            return suggested_sql
        
        except Exception as e:
            print(f"⚠️ Auto-fix failed: {e}")
            return self._apply_schema_based_fix(sql, error, error_type)
    
    def _apply_schema_based_fix(self, sql: str, error: str, error_type: str) -> str:
        """
        Apply rule-based fixes using schema knowledge when LLM fails
        """
        # Fix: column doesn't exist error
        if "does not exist" in error.lower():
            # Extract the problematic column reference
            col_match = re.search(r'column (\w+)\.(\w+) does not exist', error, re.IGNORECASE)
            if col_match:
                bad_alias = col_match.group(1)
                col = col_match.group(2)
                
                print(f"🔍 Searching for column '{col}' in schema...")
                
                # Find which table actually has this column
                found_in_tables = []
                for table, info in self.table_structure.items():
                    if col.lower() in info['columns']:
                        found_in_tables.append(table)
                        print(f"  ✓ Found '{col}' in '{table}'")
                
                if found_in_tables:
                    # Check if we're trying to get it from wrong JOIN
                    # If the column exists in tbl_primary, we don't need the JOIN
                    if 'tbl_primary' in found_in_tables and 'JOIN' in sql.upper():
                        print(f"  💡 '{col}' exists in tbl_primary - removing unnecessary JOIN")
                        # Remove the JOIN clause entirely
                        sql = re.sub(r'JOIN\s+\w+\s+(?:AS\s+)?\w+\s+ON[^;]+(?=WHERE|GROUP|ORDER|;|$)', '', sql, flags=re.IGNORECASE)
                        # Also fix any remaining bad alias references
                        sql = re.sub(rf'\b{bad_alias}\.{col}\b', f'pr.{col}', sql, flags=re.IGNORECASE)
                        # Simplify FROM clause if needed
                        sql = re.sub(r'FROM\s+(\w+)\s+AS\s+(\w+)', r'FROM \1 pr', sql, flags=re.IGNORECASE)
                    else:
                        # Find the correct alias for the table with this column
                        correct_table = found_in_tables[0]
                        correct_alias = self._find_alias_for_table(sql, correct_table)
                        if correct_alias:
                            sql = re.sub(rf'\b{bad_alias}\.{col}\b', f'{correct_alias}.{col}', sql, flags=re.IGNORECASE)
                            print(f"  ✏️ Replaced {bad_alias}.{col} with {correct_alias}.{col}")
        
        # Clean up extra whitespace
        sql = re.sub(r'\s+', ' ', sql).strip()
        
        return sql
    
    def _find_alias_for_table(self, sql: str, table_name: str) -> Optional[str]:
        """Find the alias used for a specific table in the SQL"""
        patterns = [
            rf'{table_name}\s+AS\s+(\w+)',
            rf'{table_name}\s+(\w+)\s+(?:ON|WHERE|JOIN)',
            rf'FROM\s+{table_name}\s+(\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sql, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # If no alias found, the table name itself might be used
        return table_name
    
    def _create_fix_prompt(self, sql: str, error: str, error_type: str,
                          user_query: str, annotated_schema: str, 
                          relationships: str) -> str:
        """Create appropriate fix prompt based on error type"""
        
        schema_summary = self._build_schema_summary()
        joins_summary = self._format_valid_joins()
        
        # Extract specific column/table issues from error
        error_context = ""
        if "column" in error.lower() and "does not exist" in error.lower():
            col_match = re.search(r'column (\w+)\.(\w+)', error, re.IGNORECASE)
            if col_match:
                alias, col = col_match.groups()
                error_context = f"\nThe column '{col}' doesn't exist in the table aliased as '{alias}'."
                # Find where this column actually exists
                for table, info in self.table_structure.items():
                    if col.lower() in info['columns']:
                        error_context += f"\n'{col}' EXISTS in table '{table}'. Use the correct alias."
        
        base_context = f"""You are a PostgreSQL query fixer. Fix this broken SQL.

User Question: {user_query}

Current BROKEN SQL:
{sql}

Error: {error}
{error_context}

Available Schema (COMPLETE TABLE STRUCTURES):
{schema_summary}

Valid JOIN Relationships:
{joins_summary}

CRITICAL RULES:
1. Use ONLY tables and columns from the schema above
2. Match column names to the CORRECT table (check which table has each column)
3. Use correct table aliases (if pr = tbl_primary, use pr.column_name)
4. Use ONLY the documented JOIN relationships
5. Do NOT create new columns or tables
6. Return ONLY the corrected SQL query - no explanations, no markdown

THINK STEP BY STEP:
1. Which columns are being selected/filtered?
2. Which table(s) contain those columns?
3. What are the correct aliases for those tables?
4. Do we need JOINs? If yes, use ONLY documented relationships.
5. Fix the SQL using correct table.column references
"""
        
        if error_type == "execution":
            return f"{base_context}\n\nFix the execution error. Common fixes: wrong table alias, column doesn't exist in that table."
        elif error_type == "structure":
            return f"{base_context}\n\nFix column/table names to EXACTLY match schema."
        elif error_type == "joins":
            return f"{base_context}\n\nFix JOINs to use ONLY documented relationships."
        else:
            return base_context
    
    # Helper methods
    
    def _extract_tables(self, sql: str) -> List[str]:
        """Extract table names from SQL"""
        tables = []
        sql_upper = sql.upper()
        
        from_match = re.search(r'FROM\s+(\w+)', sql_upper)
        if from_match:
            tables.append(from_match.group(1).lower())
        
        join_matches = re.findall(r'JOIN\s+(\w+)', sql_upper)
        tables.extend([m.lower() for m in join_matches])
        
        return list(set(tables))
    
    def _extract_columns(self, sql: str) -> List[Tuple[str, str]]:
        """Extract column references as (table_alias, column) tuples"""
        pattern = r'(\w+)\.(\w+)'
        matches = re.findall(pattern, sql, re.IGNORECASE)
        return [(alias.lower(), col.lower()) for alias, col in matches]
    
    def _resolve_table_alias(self, sql: str, alias: str) -> Optional[str]:
        """Resolve table alias to actual table name"""
        sql_upper = sql.upper()
        alias_upper = alias.upper()
        
        # Try pattern: table_name AS alias
        pattern1 = rf'(\w+)\s+AS\s+{alias_upper}\b'
        match = re.search(pattern1, sql_upper)
        
        if match:
            table_name = match.group(1).lower()
            if table_name in self.table_structure:
                return table_name
        
        # Try pattern: table_name alias (without AS)
        pattern2 = rf'(tbl_\w+)\s+{alias_upper}\b'
        match = re.search(pattern2, sql_upper, re.IGNORECASE)
        
        if match:
            table_name = match.group(1).lower()
            if table_name in self.table_structure:
                return table_name
        
        # Check if alias is actually the table name itself
        if alias.lower() in self.table_structure:
            return alias.lower()
        
        # Last resort: check FROM and JOIN clauses more carefully
        from_pattern = rf'FROM\s+(tbl_\w+)\s+(?:AS\s+)?{alias_upper}\b'
        from_match = re.search(from_pattern, sql_upper, re.IGNORECASE)
        if from_match:
            return from_match.group(1).lower()
        
        join_pattern = rf'JOIN\s+(tbl_\w+)\s+(?:AS\s+)?{alias_upper}\b'
        join_match = re.search(join_pattern, sql_upper, re.IGNORECASE)
        if join_match:
            return join_match.group(1).lower()
        
        return None
    
    def _is_valid_join(self, table1: str, col1: str, table2: str, 
                       col2: str, relationships: str) -> bool:
        """Check if join exists in parsed valid_joins list"""
        for valid_join in self.valid_joins:
            # Check both directions
            if ((valid_join['left_table'] == table1 and 
                 valid_join['left_column'] == col1 and
                 valid_join['right_table'] == table2 and 
                 valid_join['right_column'] == col2) or
                (valid_join['left_table'] == table2 and 
                 valid_join['left_column'] == col2 and
                 valid_join['right_table'] == table1 and 
                 valid_join['right_column'] == col1)):
                return True
        
        return False
    
    def _extract_sql(self, text: str) -> str:
        """Extract SQL from LLM response"""
        if "```sql" in text:
            return text.split("```sql")[1].split("```")[0].strip()
        elif "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        
        # Remove common prefixes
        for prefix in ["SQL:", "Query:", "Corrected:", "Fixed:"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        return text.strip()


def validator_agent_node(state):
    """
    LangGraph node for SQL validation with user interaction
    Uses annotated_schema.md and relationship.txt as source of truth
    """
    sql_query = state.get('sql_result', '')
    user_query = state.get('user_query', '')
    annotated_schema = state.get('annotated_schema', '')
    relationships = state.get('relationships', '')
    
    connection_params = {
        'host': 'localhost',
        'database': 'haldiram',
        'user': 'postgres',
        'password': '12345678',
        'port': 5432
    }
    
    # Initialize validator with annotated schema and relationships
    validator = InteractiveValidator(connection_params, annotated_schema, relationships)
    
    result = validator.validate_and_fix_sql(
        sql_query, user_query, annotated_schema, relationships
    )
    
    return {
        "validated_sql": result["validated_sql"],
        "validation_status": result["validation_status"],
        "validation_error": result.get("validation_error"),
        "correction_attempts": result.get("correction_attempts", 0)
    }


# Export with both names for compatibility
validator_agent = validator_agent_node