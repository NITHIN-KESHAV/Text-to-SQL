from typing import TypedDict, Optional, List, Any, Dict, Union
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
import re

# Local imports
from .DatabaseManager import DatabaseManager
from .LLMManager import LLMManager


class AgentState(TypedDict):
    """TypedDict to maintain state across nodes in the LangGraph workflow."""
    user_query: str
    schema: Optional[str]
    reasoning: Optional[str]
    generated_sql: Optional[str]
    is_valid: Optional[bool]
    validation_message: Optional[str]
    results: Optional[List[Any]]
    column_types: Optional[List[Any]]
    error: Optional[str]
    visualization_type: Optional[str]
    visualization_data: Optional[Dict[str, Any]]


class LangGraphAgent:
    """
    SQL Agent implementation using LangGraph for workflow management.
    """
    
    def __init__(self, debug=False):
        """Initialize the LangGraph SQL Agent with required components."""
        # Initialize dependencies
        self.db_manager = DatabaseManager()
        self.llm_manager = LLMManager(use_huggingface=True, debug=debug)
        self.debug = debug
        
        # Build and compile the workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
        
        # Cache for schema information
        self._schema_cache = None
        self._tables_cache = None
        self._columns_cache = None
    
    def _build_workflow(self) -> StateGraph:
        """Construct the LangGraph workflow defining the agent's behavior."""
        # Create a new graph with the AgentState as the state type
        workflow = StateGraph(AgentState)
        
        # Register nodes
        workflow.add_node("schema_reasoning", self.schema_reasoning_node)
        workflow.add_node("sql_generation", self.sql_generation_node)
        workflow.add_node("sql_validation", self.sql_validation_node)
        workflow.add_node("sql_execution", self.sql_execution_node)
        workflow.add_node("column_type", self.column_type_node)
        workflow.add_node("error_handling", self.error_node)
        workflow.add_node("visualization", self.visualization_node)
        
        # Define routing logic
        def sql_validation_router(state: AgentState):
            """Router after SQL validation"""
            if state.get("is_valid"):
                return "sql_execution"
            else:
                return "error_handling"
        
        def execution_router(state: AgentState):
            """Router after SQL execution"""
            if state.get("error") is None:
                return "column_type"
            else:
                return "error_handling"
            
        def error_router(state: AgentState):
            """Router after error handling"""
            # If error was fixed by regenerating SQL, go straight to execution
            if state.get("is_valid") and not state.get("error"):
                return "sql_execution"
            # If schema error occurred, try to regenerate SQL
            elif state.get("error") and "schema" in state.get("error").lower():
                return "sql_generation"
            else:
                return END
                
        def schema_router(state: AgentState):
            """Router after schema reasoning"""
            if state.get("error"):
                return END
            else:
                return "sql_generation"
        
        # Set up edges
        workflow.set_entry_point("schema_reasoning")
        workflow.add_conditional_edges("schema_reasoning", schema_router)
        workflow.add_edge("sql_generation", "sql_validation")
        workflow.add_conditional_edges("sql_validation", sql_validation_router)
        workflow.add_conditional_edges("sql_execution", execution_router)
        workflow.add_edge("column_type", "visualization")
        workflow.add_edge("visualization", END)
        workflow.add_conditional_edges("error_handling", error_router)
        
        return workflow
    
    def schema_reasoning_node(self, state: AgentState) -> AgentState:
        """Fetches database schema and analyzes the user query in relation to the schema."""
        if self.debug:
            print("[SCHEMA_REASONING] Extracting schema...")
            
        try:
            # Use cached schema if available, otherwise fetch it
            if self._schema_cache is None:
                schema = self.db_manager.get_schema()
                self._schema_cache = schema
                
                # Pre-compute tables and columns for faster validation
                self._tables_cache = self._extract_tables_from_schema(schema)
                self._columns_cache = self._extract_columns_from_schema(schema)
                
                if self.debug:
                    print(f"[SCHEMA_REASONING] Schema extracted and cached with {len(self._tables_cache)} tables")
            else:
                schema = self._schema_cache
                if self.debug:
                    print(f"[SCHEMA_REASONING] Using cached schema with {len(self._tables_cache)} tables")
            
            # Truncate schema if it's too long (focus on the most relevant parts)
            schema = self._truncate_schema(schema, state["user_query"], max_length=8000)
            
            # Add detailed relationship information to the schema
            schema = self._enhance_schema_with_relationships(schema)
            
            # Optionally preprocess the user query to normalize terms
            query = state["user_query"]
            # Simple preprocessing example (expand as needed)
            query = query.replace("movies", "films").replace("movie", "film")
            
            # Log the operation
            if self.debug:
                print(f"[SCHEMA_REASONING] Schema extracted successfully.")
                
            return {**state, "schema": schema, "user_query": query}
        except Exception as e:
            error_msg = f"Schema extraction failed: {str(e)}"
            if self.debug:
                print(f"[SCHEMA_REASONING] Error: {error_msg}")
            return {**state, "error": error_msg}
    
    def _truncate_schema(self, schema: str, query: str, max_length: int = 8000) -> str:
        """
        Truncate the schema to focus on the most relevant parts for the query.
        
        Args:
            schema: The full database schema
            query: The user query
            max_length: Maximum length of the schema
            
        Returns:
            A truncated version of the schema focused on relevant tables
        """
        if len(schema) <= max_length:
            return schema
        
        # Extract relevant keywords from the query
        query_words = query.lower().split()
        
        # Find tables that might be relevant to the query
        tables = []
        for line in schema.split("\n"):
            if line.strip().startswith("CREATE TABLE"):
                table_part = line.split("CREATE TABLE ")[1].split("(")[0].strip()
                table_name = table_part.strip('"\'[]()').lower()
                tables.append(table_name)
        
        # Find tables likely to be relevant to the query
        relevant_tables = []
        for table in tables:
            # Check if any word in the table name is in the query
            table_words = table.lower().split('_')
            if any(word in query_words for word in table_words):
                relevant_tables.append(table)
                
        # Also include commonly used tables if we haven't found many
        common_tables = ["film", "actor", "category", "film_actor", "film_category"]
        if len(relevant_tables) < 3:
            for table in common_tables:
                if table in tables and table not in relevant_tables:
                    relevant_tables.append(table)
        
        # Build a focused schema with just the relevant tables
        focused_schema = "-- Truncated schema focused on relevant tables\n\n"
        for table in relevant_tables:
            # Find the table definition
            start_idx = schema.find(f"CREATE TABLE {table}")
            if start_idx != -1:
                end_idx = schema.find(")", start_idx)
                if end_idx != -1:
                    # Include the table definition
                    focused_schema += schema[start_idx:end_idx+1] + ";\n\n"
        
        # Add a note about truncation
        focused_schema += "\n-- Schema truncated to focus on relevant tables. Other tables exist but are not shown."
        
        return focused_schema
    
    def _enhance_schema_with_relationships(self, schema: str) -> str:
        """
        Enhance the schema with detailed relationship information to assist with joins.
        """
        # Define key table relationships
        relationships_info = """
        -- DATABASE RELATIONSHIP GUIDE --
        
        -- CORE ENTITIES --
        1. FILMS:
           - The 'film' table is the central entity containing movie information
           - 'film_id' is the primary key used in many relationships
        
        2. CUSTOMERS:
           - The 'customer' table contains all customer information
           - 'customer_id' links to rental and payment tables
        
        3. ACTORS:
           - The 'actor' table contains actor information
           - Connected to films through 'film_actor' junction table
        
        -- KEY RELATIONSHIPS --
        1. Films to Categories:
           - film → film_category → category
           - film.film_id = film_category.film_id AND film_category.category_id = category.category_id
        
        2. Films to Actors:
           - film → film_actor → actor
           - film.film_id = film_actor.film_id AND film_actor.actor_id = actor.actor_id
        
        3. Rental Flow:
           - customer → rental → inventory → film
           - customer.customer_id = rental.customer_id AND rental.inventory_id = inventory.inventory_id AND inventory.film_id = film.film_id
        
        4. Payment Information:
           - customer → payment
           - customer.customer_id = payment.customer_id
           - Also links: payment.rental_id = rental.rental_id (for rental payments)
        
        -- COMMON QUERY PATTERNS --
        1. For film categories/genres: JOIN film, film_category, category
        2. For actor filmographies: JOIN actor, film_actor, film
        3. For rental history: JOIN customer, rental, inventory, film
        4. For payment analysis: JOIN customer, payment
        5. For store inventory: JOIN inventory, film, store
        
        -- DATE FIELDS --
        - rental.rental_date: When the rental occurred
        - rental.return_date: When the rental was returned
        - payment.payment_date: When the payment was made
        
        -- NOTES --
        - Time-based queries should use the date fields with proper extraction functions
        - When filtering on text fields, use LIKE with % wildcards (e.g., WHERE title LIKE '%love%')
        - For aggregations involving films, always join through inventory to get actual rentable copies
        """
        
        # Append the relationships info to the schema
        enhanced_schema = schema + "\n\n" + relationships_info
        
        return enhanced_schema
    
    def sql_generation_node(self, state: AgentState) -> AgentState:
        """Generates SQL from natural language using the LLM."""
        if self.debug:
            print(f"[GENERATING SQL] Generating SQL for question: {state['user_query']}")
            
        try:
            # Check if the query matches any pre-defined patterns first
            hardcoded_sql = self._check_for_common_patterns(state["user_query"])
            if hardcoded_sql:
                if self.debug:
                    print(f"[GENERATING SQL] Using predefined SQL for common pattern: \n{hardcoded_sql}")
                return {**state, "reasoning": "Using predefined SQL pattern", "generated_sql": hardcoded_sql}
        
            # First, add some common table mapping information to help with correct table names
            schema_supplement = """
            IMPORTANT TABLE INFORMATION:
            - For film categories or genres, use: film, film_category, category (NOT "film_genres" or "genres")
            - For actor information, use: actor, film_actor, film
            - For customer data, use: customer
            - For rental information, use: rental, inventory, film
            
            KEY RELATIONSHIPS:
            - film_actor.film_id = film.film_id (connects films to actors)
            - film_actor.actor_id = actor.actor_id (connects actors to films)
            - film_category.film_id = film.film_id (connects films to categories)
            - film_category.category_id = category.category_id (connects categories to films)
            - inventory.film_id = film.film_id (connects films to inventory)
            - rental.inventory_id = inventory.inventory_id (connects rentals to inventory)
            """
            
            # Add the supplement to the schema
            supplemented_schema = state["schema"] + "\n\n" + schema_supplement
            
            # First, generate reasoning about the query
            reasoning_prompt = self._build_reasoning_prompt(state["user_query"], supplemented_schema)
            reasoning = self.llm_manager.invoke(reasoning_prompt, run_name="sql_reasoning")
            
            # Then generate SQL based on reasoning
            sql_prompt = self._build_sql_prompt(state["user_query"], supplemented_schema, reasoning)
            sql = self.llm_manager.invoke(sql_prompt, run_name="sql_generation")
            
            # Clean up the generated SQL
            clean_sql = self._extract_sql(sql)
            
            if self.debug:
                print(f"[GENERATING SQL] Generated SQL: \n{clean_sql}")
                
            return {**state, "reasoning": reasoning, "generated_sql": clean_sql}
        except Exception as e:
            error_msg = f"SQL generation failed: {str(e)}"
            if self.debug:
                print(f"[GENERATING SQL] Error: {error_msg}")
            return {**state, "error": error_msg}
    
    def _check_for_common_patterns(self, query: str) -> Optional[str]:
        """Check if the query matches common patterns and return predefined SQL."""
        # Normalize the query for pattern matching
        query_lower = query.lower()
        
        # Pattern 1: List/Show N films from a category
        category_film_pattern = re.compile(r"(show|list|display|get|find).*?(\d+).*?(comedy|action|horror|drama|sci-fi|documentary|family|animation).*?(film|movie)s?", re.IGNORECASE)
        category_match = category_film_pattern.search(query_lower)
        if category_match:
            limit = int(category_match.group(2))
            category = category_match.group(3).title()  # Capitalize first letter
            return f"""
            SELECT f.title, c.name as category
            FROM film f
            JOIN film_category fc ON f.film_id = fc.film_id
            JOIN category c ON fc.category_id = c.category_id
            WHERE c.name = '{category}'
            ORDER BY f.title
            LIMIT {limit};
            """
        
        # Pattern 2: List/Show N films
        film_pattern = re.compile(r"(show|list|display|get|find).*?(\d+).*?(film|movie)s?", re.IGNORECASE)
        film_match = film_pattern.search(query_lower)
        if film_match:
            limit = int(film_match.group(2))
            return f"""
            SELECT f.title, f.description, f.release_year
            FROM film f
            ORDER BY f.title
            LIMIT {limit};
            """
            
        # Pattern 3.5: Actor in most films of a specific category (check this BEFORE general actor pattern)
        actor_category_pattern = re.compile(r"(actor|actors|who|actress).*?(most).*?(comedy|action|horror|drama|sci-fi|documentary|family|animation).*?(film|movie)s?", re.IGNORECASE)
        actor_category_match = actor_category_pattern.search(query_lower)
        if actor_category_match:
            category = actor_category_match.group(3).title()  # Capitalize first letter
            return f"""
            SELECT a.first_name, a.last_name, COUNT(fa.film_id) as film_count
            FROM actor a
            JOIN film_actor fa ON a.actor_id = fa.actor_id
            JOIN film_category fc ON fa.film_id = fc.film_id
            JOIN category c ON fc.category_id = c.category_id
            WHERE c.name = '{category}'
            GROUP BY a.actor_id, a.first_name, a.last_name
            ORDER BY film_count DESC
            LIMIT 10;
            """
        
        # Pattern 3: Actor in most films (general, without category)
        actor_pattern = re.compile(r"(actor|actors|who).*?(most).*?(film|movie)s?", re.IGNORECASE)
        if actor_pattern.search(query_lower):
            return """
            SELECT a.first_name, a.last_name, COUNT(fa.film_id) as film_count
            FROM actor a
            JOIN film_actor fa ON a.actor_id = fa.actor_id
            GROUP BY a.actor_id, a.first_name, a.last_name
            ORDER BY film_count DESC
            LIMIT 10;
            """
        
        # Pattern 4: Most popular/rented films
        popular_pattern = re.compile(r"(most).*?(popular|rented|watched).*?(film|movie)s?", re.IGNORECASE)
        if popular_pattern.search(query_lower):
            return """
            SELECT f.title, COUNT(r.rental_id) as rental_count
            FROM film f
            JOIN inventory i ON f.film_id = i.film_id
            JOIN rental r ON i.inventory_id = r.inventory_id
            GROUP BY f.film_id, f.title
            ORDER BY rental_count DESC
            LIMIT 10;
            """
            
        # Pattern 5: Customer spending queries
        customer_spending_pattern = re.compile(r"(customer|customers).*?(spent|spend|spending|paid|pay|paying).*?(most|money|amount)", re.IGNORECASE)
        if customer_spending_pattern.search(query_lower):
            return """
            SELECT c.customer_id, c.first_name, c.last_name, SUM(p.amount) as total_spent
            FROM customer c
            JOIN payment p ON c.customer_id = p.customer_id
            GROUP BY c.customer_id, c.first_name, c.last_name
            ORDER BY total_spent DESC
            LIMIT 10;
            """
            
        # Pattern 6: Average film length/duration
        length_pattern = re.compile(r"(average|avg|mean).*?(length|duration|runtime).*?(film|movie|category|genre)", re.IGNORECASE)
        category_in_length_query = re.search(r"(comedy|action|horror|drama|sci-fi|documentary|family|animation)", query_lower)
        
        if length_pattern.search(query_lower):
            if category_in_length_query:
                category = category_in_length_query.group(1).title()
                return f"""
                SELECT c.name as category, AVG(f.length) as avg_length
                FROM film f
                JOIN film_category fc ON f.film_id = fc.film_id
                JOIN category c ON fc.category_id = c.category_id
                WHERE c.name = '{category}'
                GROUP BY c.name;
                """
            else:
                return """
                SELECT AVG(length) as average_length
                FROM film;
                """
                
        # Pattern 7: Rental duration queries
        rental_duration_pattern = re.compile(r"(average|avg|mean).*?(rental|rented).*?(duration|time|period)", re.IGNORECASE)
        if rental_duration_pattern.search(query_lower):
            if category_in_length_query:
                category = category_in_length_query.group(1).title()
                return f"""
                SELECT c.name, AVG(f.rental_duration) as avg_rental_duration
                FROM film f
                JOIN film_category fc ON f.film_id = fc.film_id
                JOIN category c ON fc.category_id = c.category_id
                WHERE c.name = '{category}'
                GROUP BY c.name;
                """
            else:
                return """
                SELECT AVG(rental_duration) as average_rental_duration
                FROM film;
                """
                
        # Pattern 8: Films with specific text in title
        title_text_pattern = re.compile(r"(film|movie|films|movies).*?(with|contain|containing).*?['\"](.*?)['\"].*?(title|name)", re.IGNORECASE)
        title_match = title_text_pattern.search(query_lower)
        if title_match:
            search_text = title_match.group(3)
            return f"""
            SELECT title, description
            FROM film
            WHERE LOWER(title) LIKE '%{search_text.lower()}%'
            ORDER BY title
            LIMIT 10;
            """
            
        # Pattern 9: Categories with most films
        category_count_pattern = re.compile(r"(category|categories|genre|genres).*?(most|popular|common)", re.IGNORECASE)
        if category_count_pattern.search(query_lower):
            return """
            SELECT c.name as category, COUNT(fc.film_id) as film_count
            FROM category c
            JOIN film_category fc ON c.category_id = fc.category_id
            GROUP BY c.name
            ORDER BY film_count DESC;
            """
            
        # Pattern 10: Films never rented
        never_rented_pattern = re.compile(r"(film|movie|films|movies).*?(never|not).*?(rented|borrowed|watched)", re.IGNORECASE)
        if never_rented_pattern.search(query_lower):
            return """
            SELECT f.title
            FROM film f
            LEFT JOIN inventory i ON f.film_id = i.film_id
            LEFT JOIN rental r ON i.inventory_id = r.inventory_id
            WHERE r.rental_id IS NULL
            ORDER BY f.title
            LIMIT 20;
            """
            
        # Pattern 11: Rentals per time period (month/year)
        time_pattern = re.compile(r"(rentals?|rented).*?(per|by|in|during).*?(month|year|day|week).*?(\d{4})?", re.IGNORECASE)
        time_match = time_pattern.search(query_lower)
        if time_match:
            time_unit = time_match.group(3).lower()
            year = time_match.group(4) if time_match.group(4) else "2005"
            
            # For monthly breakdown
            if time_unit == "month":
                return f"""
                SELECT 
                    EXTRACT(MONTH FROM r.rental_date) as month,
                    COUNT(*) as rental_count
                FROM rental r
                WHERE EXTRACT(YEAR FROM r.rental_date) = {year}
                GROUP BY EXTRACT(MONTH FROM r.rental_date)
                ORDER BY month;
                """
            # For yearly breakdown
            elif time_unit == "year":
                return """
                SELECT 
                    EXTRACT(YEAR FROM r.rental_date) as year,
                    COUNT(*) as rental_count
                FROM rental r
                GROUP BY EXTRACT(YEAR FROM r.rental_date)
                ORDER BY year;
                """
            # For weekly breakdown
            elif time_unit == "week":
                return f"""
                SELECT 
                    EXTRACT(WEEK FROM r.rental_date) as week,
                    COUNT(*) as rental_count
                FROM rental r
                WHERE EXTRACT(YEAR FROM r.rental_date) = {year}
                GROUP BY EXTRACT(WEEK FROM r.rental_date)
                ORDER BY week;
                """
        
        # Pattern 12: Film length stats
        length_stats_pattern = re.compile(r"(common|average|typical|most).*?(length|duration|runtime).*?(film|movie)", re.IGNORECASE)
        if length_stats_pattern.search(query_lower):
            # If looking for most common length specifically
            if "common" in query_lower or "most" in query_lower:
                return """
                WITH length_counts AS (
                    SELECT 
                        length,
                        COUNT(*) as count
                    FROM film
                    GROUP BY length
                )
                SELECT 
                    length,
                    count
                FROM length_counts
                ORDER BY count DESC
                LIMIT 1;
                """
            # If looking for average length
            else:
                return """
                SELECT AVG(length) as average_length
                FROM film;
                """
                
        # Pattern 13: Staff rental processing
        staff_pattern = re.compile(r"(staff|employee).*?(process|handle).*?(most|rental)", re.IGNORECASE)
        if staff_pattern.search(query_lower) or ("staff" in query_lower and "rental" in query_lower):
            return """
            SELECT 
                s.staff_id,
                s.first_name,
                s.last_name,
                COUNT(r.rental_id) as rental_count
            FROM staff s
            JOIN rental r ON s.staff_id = r.staff_id
            GROUP BY s.staff_id, s.first_name, s.last_name
            ORDER BY rental_count DESC
            LIMIT 5;
            """
            
        # No pattern matched
        return None
    
    def sql_validation_node(self, state: AgentState) -> AgentState:
        """Validates the generated SQL query."""
        if self.debug:
            print("[VALIDATING SQL] Validating SQL query...")
            
        try:
            # Extract available tables from schema
            schema_text = state["schema"]
            available_tables = self._tables_cache or self._extract_tables_from_schema(schema_text)
            available_columns = self._columns_cache or self._extract_columns_from_schema(schema_text)
            
            # Extract tables used in the query
            sql_query = state["generated_sql"]
            tables_in_query = self._extract_tables_from_query(sql_query)
            
            # Fix common SQL syntax issues first
            sql_query = self._fix_common_sql_issues(sql_query)
            
            # Validate tables
            invalid_tables = [table for table in tables_in_query if table not in available_tables]
            
            # Basic syntax validation
            syntax_errors = []
            if not sql_query.strip().lower().startswith("select"):
                syntax_errors.append("Query must start with SELECT")
            
            if ";" not in sql_query:
                # Automatically add semicolon if missing
                sql_query += ";"
                if self.debug:
                    print("[VALIDATING SQL] Added missing semicolon to query")
            
            # Check for mismatched parentheses
            if sql_query.count('(') != sql_query.count(')'):
                open_parens = sql_query.count('(')
                close_parens = sql_query.count(')')
                syntax_errors.append(f"Mismatched parentheses: {open_parens} opening vs {close_parens} closing")
                
                # Try to fix mismatched parentheses
                if open_parens > close_parens:
                    # Add missing closing parentheses
                    sql_query = sql_query.rstrip(';') + ')' * (open_parens - close_parens) + ';'
                    if self.debug:
                        print(f"[VALIDATING SQL] Added {open_parens - close_parens} missing closing parentheses")
                elif close_parens > open_parens:
                    # Remove extra closing parentheses
                    excess_close_parens = close_parens - open_parens
                    # Count the occurrences of )
                    close_paren_positions = [i for i, char in enumerate(sql_query) if char == ')']
                    # Remove the last 'excess_close_parens' occurrences
                    for _ in range(excess_close_parens):
                        if close_paren_positions:
                            pos = close_paren_positions.pop()
                            sql_query = sql_query[:pos] + sql_query[pos+1:]
                    if self.debug:
                        print(f"[VALIDATING SQL] Removed {excess_close_parens} excess closing parentheses")
            
            # Fix CTE issues or invalid nested WITH clauses
            if "WITH" in sql_query.upper() and not re.search(r'WITH\s+\w+\s+AS\s*\(', sql_query, re.IGNORECASE):
                syntax_errors.append("Invalid WITH clause syntax")
                # Try to convert WITH clauses to standard SELECTs if possible
                if self.debug:
                    print("[VALIDATING SQL] Attempting to fix WITH clause syntax")
                sql_query = self._fix_with_clause(sql_query)
            
            # Check for basic join issues
            if " JOIN " in sql_query.upper() and " ON " not in sql_query.upper():
                syntax_errors.append("JOIN without ON condition")
                # Try to fix by adding proper join conditions from schema
                sql_query = self._fix_join_conditions(sql_query, available_tables, available_columns)
                if " ON " in sql_query.upper():  # Check if fix was successful
                    if self.debug:
                        print("[VALIDATING SQL] Fixed JOIN without ON condition")
                    syntax_errors.remove("JOIN without ON condition")
            
            # Check for aggregation function issues (GROUP BY missing)
            agg_functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
            has_agg_function = any(func in sql_query.upper() for func in agg_functions)
            has_group_by = "GROUP BY" in sql_query.upper()
            
            if has_agg_function and not has_group_by and ',' in sql_query:
                # Check if there's a mix of aggregated and non-aggregated columns
                select_clause = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE | re.DOTALL)
                if select_clause:
                    columns = select_clause.group(1).split(',')
                    agg_columns = [col for col in columns if any(func in col.upper() for func in agg_functions)]
                    non_agg_columns = [col for col in columns if not any(func in col.upper() for func in agg_functions)]
                    
                    if agg_columns and non_agg_columns:
                        syntax_errors.append("Mixing aggregated and non-aggregated columns without GROUP BY")
                        
                        # Try to fix by adding GROUP BY for non-aggregated columns
                        if self.debug:
                            print("[VALIDATING SQL] Adding GROUP BY clause for non-aggregated columns")
                        
                        group_columns = ", ".join([col.strip() for col in non_agg_columns])
                        order_by_match = re.search(r'ORDER BY\s+(.*?)(\s+LIMIT|\s*;|$)', sql_query, re.IGNORECASE)
                        
                        if order_by_match:
                            # Insert GROUP BY before ORDER BY
                            order_by_pos = sql_query.upper().find("ORDER BY")
                            sql_query = f"{sql_query[:order_by_pos]}GROUP BY {group_columns} {sql_query[order_by_pos:]}"
                        else:
                            # Add GROUP BY at the end (before any LIMIT clause or semicolon)
                            limit_match = re.search(r'LIMIT\s+\d+\s*;?$', sql_query, re.IGNORECASE)
                            if limit_match:
                                limit_pos = sql_query.upper().find("LIMIT")
                                sql_query = f"{sql_query[:limit_pos]}GROUP BY {group_columns} {sql_query[limit_pos:]}"
                            else:
                                # Add before semicolon
                                sql_query = sql_query.rstrip(';') + f" GROUP BY {group_columns};"
            
            # If we have invalid tables, try to fix them
            if invalid_tables:
                # Generate correction suggestions
                correction_suggestions = {}
                common_mistakes = {
                    "films": "film",
                    "movies": "film",
                    "film_genre": "film_category",
                    "genre": "category",
                    "genres": "category",
                    "actors": "actor",
                    "customers": "customer",
                    "rentals": "rental",
                    "films_actor": "film_actor",
                    "film_actors": "film_actor",
                    "film_categories": "film_category",
                    "movie": "film",
                    "payments": "payment",
                    "inventories": "inventory",
                    "stores": "store",
                    "staffs": "staff",
                    "addresses": "address",
                    "cities": "city",
                    "countries": "country",
                    "languages": "language"
                }
                
                for table in invalid_tables:
                    # Check common mistakes
                    if table.lower() in common_mistakes and common_mistakes[table.lower()] in available_tables:
                        correction_suggestions[table] = common_mistakes[table.lower()]
                    # Check plural/singular forms
                    elif table.lower().endswith('s') and table.lower()[:-1] in available_tables:
                        correction_suggestions[table] = table.lower()[:-1]
                    # Check for tables with underscores having the wrong format
                    elif '_' in table.lower():
                        parts = table.lower().split('_')
                        possible_corrections = [
                            f"{parts[0]}_{parts[1][:-1]}" if parts[1].endswith('s') else None,
                            f"{parts[0][:-1]}_{parts[1]}" if parts[0].endswith('s') else None
                        ]
                        for correction in possible_corrections:
                            if correction and correction in available_tables:
                                correction_suggestions[table] = correction
                
                # Format correction message
                correction_msg = ""
                if correction_suggestions:
                    suggestions = [f"'{wrong}' should be '{right}'" for wrong, right in correction_suggestions.items()]
                    correction_msg = f" Suggested corrections: {'; '.join(suggestions)}"
                
                validation_message = f"Invalid tables in query: {', '.join(invalid_tables)}.{correction_msg}"
                
                # Try to fix the query automatically if we have corrections
                if correction_suggestions:
                    fixed_query = sql_query
                    for wrong, right in correction_suggestions.items():
                        fixed_query = re.sub(r'\b' + re.escape(wrong) + r'\b', right, fixed_query, flags=re.IGNORECASE)
                    
                    # Test if we fixed all the issues
                    fixed_tables = self._extract_tables_from_query(fixed_query)
                    remaining_invalid = [table for table in fixed_tables if table not in available_tables]
                    
                    if not remaining_invalid and not syntax_errors:
                        if self.debug:
                            print(f"[VALIDATING SQL] Auto-fixed query: {fixed_query}")
                        return {**state, "is_valid": True, "generated_sql": fixed_query, 
                                "validation_message": f"Query auto-corrected: {', '.join(correction_suggestions.keys())} → {', '.join(correction_suggestions.values())}"}
                
                if self.debug:
                    print(f"[VALIDATING SQL] {validation_message}")
                
                # Return invalid result if we couldn't fix the tables or have syntax errors
                if syntax_errors:
                    validation_message = f"{validation_message}\nSyntax errors: {'; '.join(syntax_errors)}"
                    
                return {**state, "is_valid": False, "validation_message": validation_message}
            
            # Return syntax errors if any
            if syntax_errors:
                validation_message = f"Syntax errors: {'; '.join(syntax_errors)}"
                if self.debug:
                    print(f"[VALIDATING SQL] {validation_message}")
                return {**state, "is_valid": False, "validation_message": validation_message}
                
            if self.debug:
                print("[VALIDATING SQL] SQL query validated successfully.")
                
            return {**state, "is_valid": True, "generated_sql": sql_query}
        except Exception as e:
            error_msg = f"SQL validation failed: {str(e)}"
            if self.debug:
                print(f"[VALIDATING SQL] Error: {error_msg}")
            return {**state, "is_valid": False, "validation_message": error_msg}
    
    def _fix_common_sql_issues(self, sql_query):
        """Fix common SQL syntax issues."""
        # Remove extra spaces
        sql_query = re.sub(r'\s+', ' ', sql_query.strip())
        
        # Make sure there's a space after commas
        sql_query = re.sub(r',(?=\S)', ', ', sql_query)
        
        # Fix incomplete CTEs (WITH clauses)
        if re.search(r'WITH\s+\w+\s+AS\s*\([^)]*$', sql_query, re.IGNORECASE):
            # If we have an unclosed CTE, close it
            sql_query = re.sub(r'(WITH\s+\w+\s+AS\s*\([^)]*)(SELECT)', r'\1) \2', sql_query, flags=re.IGNORECASE)
        
        # Remove comments
        sql_query = re.sub(r'--.*?$', '', sql_query, flags=re.MULTILINE)
        
        return sql_query
    
    def _fix_with_clause(self, sql_query):
        """Try to fix or simplify problematic WITH clauses."""
        # If the WITH clause is malformed, convert to a simple SELECT if possible
        if "WITH" in sql_query.upper():
            # Check if it's a simple WITH without proper AS (...) structure
            if not re.search(r'WITH\s+\w+\s+AS\s*\(', sql_query, re.IGNORECASE):
                # Extract the main SELECT statement if it exists
                select_match = re.search(r'SELECT\s+.*?FROM', sql_query, re.IGNORECASE | re.DOTALL)
                if select_match:
                    # Try to create a simple SELECT query
                    simplified_sql = sql_query[select_match.start():]
                    if ";" not in simplified_sql:
                        simplified_sql += ";"
                    return simplified_sql
        
        return sql_query
    
    def _fix_join_conditions(self, sql_query, available_tables, available_columns):
        """Try to fix missing JOIN conditions based on schema knowledge."""
        # Extract the FROM and JOIN parts of the query
        query_lower = sql_query.lower()
        from_match = re.search(r'from\s+(\w+)', query_lower)
        
        if not from_match:
            return sql_query  # Can't fix if we can't identify the base table
            
        base_table = from_match.group(1)
        join_matches = re.finditer(r'join\s+(\w+)(?:\s+(?:as\s+)?(\w+))?(?:\s+on\s+(.+?))?(?=\s+(?:join|where|group|order|limit|$))', query_lower)
        
        fixed_query = sql_query
        
        # Define common join conditions based on table names
        common_joins = {
            ('film', 'film_category'): 'film.film_id = film_category.film_id',
            ('film_category', 'category'): 'film_category.category_id = category.category_id',
            ('film', 'film_actor'): 'film.film_id = film_actor.film_id',
            ('film_actor', 'actor'): 'film_actor.actor_id = actor.actor_id',
            ('customer', 'rental'): 'customer.customer_id = rental.customer_id',
            ('inventory', 'rental'): 'inventory.inventory_id = rental.inventory_id',
            ('film', 'inventory'): 'film.film_id = inventory.film_id',
            ('customer', 'payment'): 'customer.customer_id = payment.customer_id',
            ('rental', 'payment'): 'rental.rental_id = payment.rental_id',
            ('store', 'inventory'): 'store.store_id = inventory.store_id',
            ('store', 'staff'): 'store.store_id = staff.store_id',
            ('staff', 'payment'): 'staff.staff_id = payment.staff_id',
            ('address', 'store'): 'address.address_id = store.address_id',
            ('city', 'address'): 'city.city_id = address.city_id',
            ('country', 'city'): 'country.country_id = city.country_id',
            ('address', 'customer'): 'address.address_id = customer.address_id',
            ('language', 'film'): 'language.language_id = film.language_id',
        }
        
        # Add reverse mappings
        for (t1, t2), condition in list(common_joins.items()):
            common_joins[(t2, t1)] = condition
        
        for match in join_matches:
            joined_table = match.group(1)
            table_alias = match.group(2)
            existing_condition = match.group(3)
            
            if not existing_condition:  # Missing ON condition
                # Find a join condition between the current base table and the joined table
                target_table = joined_table
                
                # Try to find a join condition
                join_key = (base_table, target_table)
                if join_key in common_joins:
                    join_condition = common_joins[join_key]
                    
                    # Replace this join with proper ON condition
                    join_text = match.group(0)
                    fixed_join = f"JOIN {joined_table}{' AS ' + table_alias if table_alias else ''} ON {join_condition}"
                    
                    # Update the query
                    fixed_query = fixed_query.replace(join_text, fixed_join, 1)
                    
                # Update base table for next join
                base_table = target_table
        
        return fixed_query
    
    def sql_execution_node(self, state: AgentState) -> AgentState:
        """Executes the validated SQL query against the database."""
        if self.debug:
            print("[EXECUTING SQL] Executing SQL query...")
            
        try:
            # Set a timeout for the execution
            max_execution_time = 10  # seconds
            
            # Check if the query might be too expensive (no LIMIT clause on potentially large result sets)
            sql_query = state["generated_sql"]
            if "LIMIT" not in sql_query.upper() and not self._is_aggregation_query(sql_query):
                # Add a LIMIT clause to potentially expensive queries
                sql_query = self._add_limit_clause(sql_query, 100)
                if self.debug:
                    print(f"[EXECUTING SQL] Added LIMIT clause to query: {sql_query}")
                state["generated_sql"] = sql_query
            
            # Execute the query with timeout protection
            import threading
            import time
            
            result = [None]
            error = [None]
            execution_complete = [False]
            
            def execute_with_timeout():
                try:
                    result[0] = self.db_manager.execute_query(sql_query)
                    execution_complete[0] = True
                except Exception as e:
                    error[0] = str(e)
            
            # Start execution in a separate thread
            execution_thread = threading.Thread(target=execute_with_timeout)
            execution_thread.daemon = True
            execution_thread.start()
            
            # Wait for completion or timeout
            start_time = time.time()
            while time.time() - start_time < max_execution_time and not execution_complete[0]:
                if not execution_thread.is_alive():
                    break
                time.sleep(0.1)
                
            # Check if execution completed or timed out
            if not execution_complete[0]:
                # Query timed out
                return {**state, "error": f"Query execution timed out after {max_execution_time} seconds. The query might be too complex or return too many results."}
            
            # Check for errors
            if error[0]:
                return {**state, "error": f"Query execution failed: {error[0]}"}
            
            # Successfully executed
            if self.debug:
                print(f"[EXECUTING SQL] Query executed successfully. Result: {result[0]}")
                
            return {**state, "results": result[0]}
            
        except Exception as e:
            error_msg = f"Query execution failed: {str(e)}"
            if self.debug:
                print(f"[EXECUTING SQL] Error: {error_msg}")
            return {**state, "error": error_msg}
            
    def _is_aggregation_query(self, query: str) -> bool:
        """Check if the query uses aggregation functions or GROUP BY."""
        agg_keywords = ['COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX(', 'GROUP BY', 'HAVING']
        return any(keyword in query.upper() for keyword in agg_keywords)
        
    def _add_limit_clause(self, query: str, limit: int = 100) -> str:
        """Add a LIMIT clause to a query if it doesn't already have one."""
        if "LIMIT" in query.upper():
            return query
            
        # Remove trailing semicolon if present
        query = query.rstrip(';')
        
        # Add LIMIT clause
        query = f"{query} LIMIT {limit};"
        
        return query
    
    def column_type_node(self, state: AgentState) -> AgentState:
        """Extracts column metadata from query results."""
        if self.debug:
            print("[COLUMN_TYPES] Extracting column metadata...")
            
        try:
            # This would depend on how your DB manager returns results
            # Typically you would extract column names and types here
            column_types = self.db_manager.get_last_query_column_types()
            
            if self.debug:
                print(f"[COLUMN_TYPES] Extracted column types: {column_types}")
                
            return {**state, "column_types": column_types}
        except Exception as e:
            error_msg = f"Column type extraction failed: {str(e)}"
            if self.debug:
                print(f"[COLUMN_TYPES] Error: {error_msg}")
            return {**state, "error": error_msg}
    
    def visualization_node(self, state: AgentState) -> AgentState:
        """Determines appropriate visualization for the query results."""
        if self.debug:
            print("[VISUALIZATION] Determining appropriate visualization...")
            
        try:
            # Logic to determine the best visualization based on result structure
            results = state.get("results", [])
            column_types = state.get("column_types", [])
            
            if not results or not column_types:
                return {**state, "visualization_type": "none"}
            
            # Simple visualization logic
            visualization_type = "table"  # default
            
            # Check for time series data
            date_columns = [col for col, typ in column_types if "date" in str(typ).lower()]
            numeric_columns = [col for col, typ in column_types if "int" in str(typ).lower() or "float" in str(typ).lower() or "numeric" in str(typ).lower()]
            
            if date_columns and numeric_columns:
                visualization_type = "line_chart"
            elif len(numeric_columns) >= 2:
                visualization_type = "bar_chart"
            elif len(results) <= 5 and numeric_columns:
                visualization_type = "pie_chart"
            
            if self.debug:
                print(f"[VISUALIZATION] Selected visualization type: {visualization_type}")
                
            return {**state, "visualization_type": visualization_type}
        except Exception as e:
            error_msg = f"Visualization selection failed: {str(e)}"
            if self.debug:
                print(f"[VISUALIZATION] Error: {error_msg}")
            return {**state, "error": error_msg}
    
    def error_node(self, state: AgentState) -> AgentState:
        """Handles errors that occur during processing."""
        error = state.get("error") or state.get("validation_message") or "Unknown error"
        
        if self.debug:
            print(f"[ERROR_HANDLING] Error: {error}")
        
        # Categorize error type
        error_type = "table_error" if "Invalid tables" in error else \
                    "syntax_error" if "Syntax error" in error else \
                    "timeout_error" if "timeout" in error else \
                    "generic_error"
        
        if self.debug:
            print(f"[ERROR_HANDLING] Error type: {error_type}")
        
        # Extract available tables for use in prompts
        available_tables = self._tables_cache or self._extract_tables_from_schema(state["schema"])
        important_tables = ", ".join(["film", "actor", "category", "film_actor", "film_category", 
                                     "customer", "rental", "inventory", "payment", "staff", "store"])
        
        # Construct a targeted prompt based on error type
        if error_type == "table_error":
            # Extract corrections if available
            corrections = {}
            if "Suggested corrections" in error:
                correction_parts = error.split("Suggested corrections: ")[1].split(";")
                for part in correction_parts:
                    if "should be" in part:
                        wrong, right = part.split("should be")
                        wrong = wrong.strip().strip("'")
                        right = right.strip().strip("'")
                        corrections[wrong] = right
            
            correction_info = "\n".join([f"- '{wrong}' should be '{right}'" for wrong, right in corrections.items()])
            
            prompt_template = f'''
You are a PostgreSQL expert. Fix a SQL query that uses invalid table names.

User question: {state.get("user_query")}

Original SQL with errors:
{state.get("generated_sql")}

Error: {error}

Table corrections needed:
{correction_info if corrections else "Use only valid tables and fix table names."}

Valid tables: {", ".join(sorted(available_tables))}
Important tables: {important_tables}

Key relationships:
- film_actor: connects actors to films (film_id, actor_id)
- film_category: connects films to categories (film_id, category_id)
- inventory: connects films to stores (film_id, store_id)
- rental: connects customers to inventory (customer_id, inventory_id)
- payment: tracks payments (customer_id, amount)

Generate a corrected SQL query:
'''
        
        elif error_type == "syntax_error":
            prompt_template = f'''
You are a SQL syntax expert. Fix the syntax errors in this PostgreSQL query.

User question: {state.get("user_query")}

Original SQL with errors:
{state.get("generated_sql")}

Error: {error}

Some common syntax issues to check:
- Mismatched parentheses
- Missing JOIN conditions
- Incorrect GROUP BY usage
- WITH clause syntax issues
- Invalid column references

Generate a corrected SQL query:
'''
        
        elif error_type == "timeout_error":
            prompt_template = f'''
You are a PostgreSQL optimization expert. This query timed out for being too complex.

User question: {state.get("user_query")}

Original complex SQL that timed out:
{state.get("generated_sql")}

Error: {error}

Simplify this query by:
- Reducing the number of joins
- Adding a proper LIMIT clause (e.g., LIMIT 10)
- Avoiding complex subqueries
- Using simpler filtering conditions

Generate a simplified SQL query:
'''
        
        else:  # generic_error
            prompt_template = f'''
You are a PostgreSQL expert. Fix this SQL query to address the error.

User question: {state.get("user_query")}

Original SQL with errors:
{state.get("generated_sql")}

Error: {error}

Valid tables: {", ".join(sorted(available_tables))}
Important tables: {important_tables}

Generate a corrected SQL query that accurately answers the question:
'''
        
        try:
            if self.debug:
                print(f"[ERROR_HANDLING] Attempting to regenerate SQL")
            
            # Create a chat prompt
            correction_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a SQL expert who fixes SQL queries. Output only the corrected SQL query."),
                ("human", prompt_template)
            ])
            
            # Generate corrected SQL
            corrected_sql = self.llm_manager.invoke(correction_prompt, run_name="sql_error_correction")
            
            # Clean up the generated SQL
            clean_sql = self._extract_sql(corrected_sql)
            
            if self.debug:
                print(f"[ERROR_HANDLING] Regenerated SQL: {clean_sql}")
            
            # Validate corrected query
            if clean_sql and clean_sql.strip().lower().startswith("select"):
                # Check for invalid tables in the corrected query
                tables_in_query = self._extract_tables_from_query(clean_sql)
                invalid_tables = [table for table in tables_in_query if table.lower() not in available_tables]
                
                if not invalid_tables:
                    # The regenerated query looks valid
                    return {
                        **state, 
                        "error": None,
                        "validation_message": None,
                        "is_valid": True,
                        "generated_sql": clean_sql
                    }
                else:
                    if self.debug:
                        print(f"[ERROR_HANDLING] Corrected SQL still has invalid tables: {invalid_tables}")
            
            # If we reached here, try a super simple final attempt
            if self.debug:
                print(f"[ERROR_HANDLING] Making final simplified attempt")
            
            simple_prompt = ChatPromptTemplate.from_messages([
                ("system", "Generate an extremely simple SQL query using only valid tables. Output SQL only."),
                ("human", f'''
Question: {state.get("user_query")}
Use ONLY these tables: {", ".join([t for t in available_tables if t in ["film", "actor", "category", "film_actor", "film_category", "customer", "rental", "inventory", "payment"]])}
Add LIMIT 10 to the query. Keep it extremely simple.
''')
            ])
            
            last_attempt = self.llm_manager.invoke(simple_prompt, run_name="final_simple_correction")
            clean_last = self._extract_sql(last_attempt)
            
            if self.debug:
                print(f"[ERROR_HANDLING] Final attempt SQL: {clean_last}")
            
            final_tables = self._extract_tables_from_query(clean_last)
            final_invalid = [t for t in final_tables if t.lower() not in available_tables]
            
            if not final_invalid and clean_last.strip().lower().startswith("select"):
                return {
                    **state, 
                    "error": None,
                    "validation_message": None,
                    "is_valid": True,
                    "generated_sql": clean_last
                }
                
        except Exception as regen_error:
            if self.debug:
                print(f"[ERROR_HANDLING] Error during SQL regeneration: {str(regen_error)}")
        
        return state
    
    # Helper methods
    def _build_reasoning_prompt(self, question: str, schema: str) -> str:
        """Builds a prompt for reasoning about the user's question."""
        return ChatPromptTemplate.from_messages([
            ("system", '''You are a database expert analyzing a question to determine how to query a PostgreSQL database.
Your task is to analyze the user's question, identify the required tables and relationships,
and explain how to structure a valid SQL query that will answer the question.

Follow this structured approach:
1. Identify the key entities/tables needed to answer the question
2. Identify any relevant conditions or filters
3. Determine if aggregation, grouping, or ordering is needed
4. Consider any joins required between tables
5. Check if any computed values or transformations are needed

You will be provided with a database schema that describes the available tables and columns.
Carefully analyze this schema before suggesting any tables or columns.'''),
            ("human", f'''
User question: {question}

Database schema:
{schema}

Before I write SQL code, I'll analyze what this question is asking:

1. What specific tables will I need?
2. What relationships (joins) are needed between these tables?
3. What columns need to be selected or calculated?
4. What conditions or filters should be applied?
5. Do I need any aggregation, grouping, or ordering?

Reasoning:
''')
        ])
    
    def _build_sql_prompt(self, question: str, schema: str, reasoning: str) -> str:
        """Builds a prompt for generating SQL from the user's question."""
        return ChatPromptTemplate.from_messages([
            ("system", '''You are a database expert that converts natural language questions into PostgreSQL queries.
Follow these rules precisely:
1. Use ONLY the tables and columns that exist in the provided schema
2. Always use singular table names (e.g., "film" not "films")
3. Do NOT use CTEs (WITH clause) unless absolutely necessary
4. Do NOT create aliases for tables unless needed for joins
5. Verify all tables and joins are valid before returning SQL
6. Keep queries simple - use the most direct approach
7. Always add a LIMIT if returning multiple rows
8. Correct format: SELECT columns FROM table_a JOIN table_b ON condition WHERE filter;
9. Avoid subqueries when a direct join will work
10. DO NOT reference tables that don't exist in the schema

COMMON MISTAKES TO AVOID:
- Do not use "film_genre" (correct: "film_category" and "category")
- Do not use "rental_date" without checking it exists in the table
- Do not use plural table names like "films" (use "film" instead)
- DO NOT create your own tables/views (e.g., "ActionFilms" or "TopFiveFilms")
'''),
            ("human", f'''
User question: {question}

Database schema:
{schema}

Reasoning about the question:
{reasoning}

Here are a few examples of correct SQL queries for this database:

Example 1: "Show me comedy films"
```sql
SELECT f.title 
FROM film f
JOIN film_category fc ON f.film_id = fc.film_id
JOIN category c ON fc.category_id = c.category_id
WHERE c.name = 'Comedy'
LIMIT 10;
```

Example 2: "List actors in more than 20 films"
```sql
SELECT a.actor_id, a.first_name, a.last_name, COUNT(fa.film_id) as film_count
FROM actor a
JOIN film_actor fa ON a.actor_id = fa.actor_id
GROUP BY a.actor_id, a.first_name, a.last_name
HAVING COUNT(fa.film_id) > 20
ORDER BY film_count DESC;
```

Example 3: "Show me the top 5 most rented films"
```sql
SELECT f.title, COUNT(r.rental_id) as rental_count
FROM film f
JOIN inventory i ON f.film_id = i.film_id
JOIN rental r ON i.inventory_id = r.inventory_id
GROUP BY f.title
ORDER BY rental_count DESC
LIMIT 5;
```

Now, write a PostgreSQL SQL query for this question: {question}
Only provide the SQL query, nothing else. The query should be syntactically correct and executable.
''')
        ])
    
    def _extract_sql(self, text: str) -> str:
        """Extracts and cleans the SQL query from the LLM response."""
        # Attempt several extraction patterns
        
        # 1. Look for SQL between triple backticks
        sql_code_blocks = re.findall(r"```(?:sql)?(.*?)```", text, re.DOTALL)
        if sql_code_blocks:
            # Join all code blocks in case there are multiple
            complete_sql = "\n".join([block.strip() for block in sql_code_blocks])
            return complete_sql
            
        # 2. Find SQL between SQL keywords
        complete_query = None
        if ("SELECT" in text.upper() or "WITH" in text.upper()):
            # First check if there's a complete SELECT query with semicolon
            select_match = re.search(r"(SELECT\s+[\s\S]+?);", text, re.IGNORECASE)
            if select_match:
                complete_query = select_match.group(1).strip() + ";"
            
            # Check for WITH clause
            with_match = re.search(r"(WITH\s+[\s\S]+?SELECT[\s\S]+?);", text, re.IGNORECASE)
            if with_match:
                complete_query = with_match.group(1).strip() + ";"
            
            # If no complete query found, try to extract all lines that look like SQL
            if not complete_query:
                lines = text.split('\n')
                sql_lines = []
                in_sql = False
                
                for line in lines:
                    line = line.strip()
                    # Check for SQL keywords
                    if (any(keyword in line.upper() for keyword in ["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING", "JOIN", "LIMIT"]) 
                            or in_sql and line and not line.startswith(('```', '/*', '--', '#'))):
                        if "SELECT" in line.upper():
                            in_sql = True
                        if in_sql:
                            sql_lines.append(line)
                
                if sql_lines:
                    # Join and clean up SQL lines
                    candidate_sql = " ".join(sql_lines)
                    # Make sure we have a complete query
                    if "SELECT" in candidate_sql.upper() and "FROM" in candidate_sql.upper():
                        complete_query = candidate_sql
                        if ";" not in complete_query:
                            complete_query += ";"
        
        # 3. Return the complete query or cleaned text
        if complete_query:
            return self._clean_sql(complete_query)
        
        # 4. Just return the cleaned text as a last resort
        return self._clean_sql(text.strip())
        
    def _clean_sql(self, sql: str) -> str:
        """Clean up the SQL query."""
        # Remove any markdown formatting
        sql = re.sub(r'```sql|```', '', sql).strip()
        
        # Remove explanatory text before SELECT or after semicolon
        if "SELECT" in sql.upper():
            before_select = sql.upper().split("SELECT")[0]
            # If there's text before SELECT, remove it
            if before_select.strip():
                sql = "SELECT" + sql.upper().split("SELECT", 1)[1]
        
        # Remove anything after the last semicolon
        if ";" in sql:
            sql = sql.split(";")[0] + ";"
            
        # Fix spacing around punctuation
        sql = re.sub(r'\s+', ' ', sql)
        sql = re.sub(r'\(\s+', '(', sql)
        sql = re.sub(r'\s+\)', ')', sql)
        sql = re.sub(r'\s+,', ',', sql)
        sql = re.sub(r',(?! )', ', ', sql)
        
        # Fix common syntax errors
        # Replace double quotes on table/column identifiers with proper PostgreSQL quotes
        sql = re.sub(r'"(\w+)"', r'"\1"', sql)
        
        return sql.strip()
    
    def _extract_tables_from_schema(self, schema: str) -> List[str]:
        """Extracts available table names from the schema."""
        # This is a simplified implementation
        # In practice, you would parse the schema more carefully
        tables = []
        for line in schema.split('\n'):
            if line.strip().startswith("CREATE TABLE"):
                table_name = line.split("CREATE TABLE")[1].split("(")[0].strip()
                tables.append(table_name)
        return tables
    
    def _extract_tables_from_query(self, query: str) -> List[str]:
        """Extracts table names used in the SQL query."""
        # This is a simplified implementation
        # In practice, you would use a proper SQL parser
        import re
        # Find tables after FROM and JOIN
        from_tables = re.findall(r"FROM\s+(\w+)", query, re.IGNORECASE)
        join_tables = re.findall(r"JOIN\s+(\w+)", query, re.IGNORECASE)
        return list(set(from_tables + join_tables))
    
    def _extract_columns_from_schema(self, schema: str) -> Dict[str, List[str]]:
        """Extract column information from the schema grouped by table."""
        columns_by_table = {}
        current_table = None
        
        for line in schema.split('\n'):
            # Find table definition
            if line.strip().startswith("CREATE TABLE"):
                table_part = line.split("CREATE TABLE")[1].split("(")[0].strip()
                current_table = table_part.strip('"\'[]()').lower()
                columns_by_table[current_table] = []
            # Extract columns while inside a table definition
            elif current_table and "(" in line and not line.strip().startswith("CREATE"):
                # Parse out column name
                parts = line.strip().split()
                if parts and "FOREIGN KEY" not in line and "PRIMARY KEY" not in line:
                    col_name = parts[0].strip(',')
                    columns_by_table[current_table].append(col_name.lower())
        
        return columns_by_table
    
    def process_query(self, question: str) -> Dict[str, Any]:
        """
        Process a natural language query through the LangGraph workflow.
        
        Args:
            question: The natural language question to process
            
        Returns:
            Dict with query results, SQL, and other state information
        """
        # Initialize the state with just the user query
        initial_state = {"user_query": question}
        
        # Invoke the workflow
        try:
            final_state = self.app.invoke(initial_state)
            return final_state
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            if self.debug:
                print(f"[PROCESS_QUERY] Error: {error_msg}")
            return {"user_query": question, "error": error_msg} 