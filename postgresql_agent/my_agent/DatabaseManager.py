import os
import psycopg2
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase

class DatabaseManager:
    """
    DatabaseManager class to connect to and query a PostgreSQL database.
    """
    
    def __init__(self):
        """
        Initialize the DatabaseManager with connection parameters from environment variables.
        """
        # Load environment variables from .env file if it exists
        load_dotenv()
        
        # Get database connection parameters from environment variables
        self.db_name = os.environ.get("DB_NAME", "pagila")
        self.db_user = os.environ.get("DB_USER", "postgres")
        self.db_password = os.environ.get("DB_PASSWORD", "")
        self.db_host = os.environ.get("DB_HOST", "localhost")
        self.db_port = os.environ.get("DB_PORT", "5432")
        
        # Initialize connection
        self.conn = None
        self.connect_to_db()
    
    def connect_to_db(self):
        """Connect to the PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port
            )
            print("Connected to the database successfully!")
            
            # List tables for debugging
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;")
                tables = [row[0] for row in cursor.fetchall()]
                print(f"Tables in the database: {tables}")
                
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            raise
    
    def get_schema(self):
        """Get the database schema including tables, columns, and sample data."""
        try:
            schema_text = ""
            
            # Get all tables
            tables_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name;
            """
            tables_result = self.execute_query(tables_query)
            
            # For each table, get its columns
            for table_row in tables_result:
                table_name = table_row[0]
                
                # Get columns for this table
                columns_query = f"""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = '{table_name}'
                ORDER BY ordinal_position;
                """
                columns_result = self.execute_query(columns_query)
                
                # Start building the CREATE TABLE statement
                create_table = f"CREATE TABLE {table_name} (\n"
                
                # Add column definitions
                column_defs = []
                for col in columns_result:
                    col_name, col_type, nullable, default = col
                    col_def = f"    {col_name} {col_type}"
                    
                    if default is not None:
                        col_def += f" DEFAULT {default}"
                    
                    if nullable == 'NO':
                        col_def += " NOT NULL"
                        
                    column_defs.append(col_def)
                
                # Get primary key constraints
                pk_query = f"""
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu 
                ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_schema = 'public' 
                AND tc.table_name = '{table_name}'
                AND tc.constraint_type = 'PRIMARY KEY';
                """
                pk_result = self.execute_query(pk_query)
                
                if pk_result:
                    pk_columns = [row[0] for row in pk_result]
                    column_defs.append(f"    PRIMARY KEY ({', '.join(pk_columns)})")
                
                # Get foreign key constraints
                fk_query = f"""
                SELECT
                    kcu.column_name,
                    ccu.table_name as foreign_table_name,
                    ccu.column_name as foreign_column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                    ON tc.constraint_name = ccu.constraint_name
                WHERE tc.table_schema = 'public'
                AND tc.table_name = '{table_name}'
                AND tc.constraint_type = 'FOREIGN KEY';
                """
                fk_result = self.execute_query(fk_query)
                
                for fk in fk_result:
                    col_name, foreign_table, foreign_col = fk
                    column_defs.append(f"    FOREIGN KEY ({col_name}) REFERENCES {foreign_table} ({foreign_col})")
                
                # Finalize the CREATE TABLE statement
                create_table += ",\n".join(column_defs) + "\n);"
                schema_text += create_table + "\n\n"
                
                # Get sample data
                try:
                    sample_query = f"SELECT * FROM {table_name} LIMIT 3;"
                    sample_data = self.execute_query(sample_query)
                    
                    if sample_data:
                        # Get column names
                        col_names = [col[0] for col in columns_result]
                        
                        # Format sample data as comment
                        sample_comment = f"/*\n3 rows from {table_name} table:\n"
                        sample_comment += "\t".join(col_names) + "\n"
                        
                        for row in sample_data:
                            sample_comment += "\t".join([str(col) for col in row]) + "\n"
                        
                        sample_comment += "*/\n"
                        
                        # Add sample data to schema
                        schema_text += sample_comment + "\n"
                except Exception as e:
                    print(f"Error getting sample data for table {table_name}: {str(e)}")
            
            return schema_text
            
        except Exception as e:
            print(f"Error getting schema: {str(e)}")
            return f"Error: {str(e)}"
    
    def execute_query(self, query):
        """Execute a SQL query and return the results."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query)
                
                # Store cursor description for later column type retrieval
                self._last_cursor_description = cursor.description
                
                # Return results or None for non-SELECT queries
                if cursor.description:
                    return cursor.fetchall()
                else:
                    self.conn.commit()
                    return None
        except Exception as e:
            self.conn.rollback()
            raise Exception(f"Query execution failed: {str(e)}")
    
    def get_last_query_column_types(self):
        """
        Returns column metadata from the most recently executed query.
        Returns a list of (column_name, type_code) tuples.
        """
        if not hasattr(self, '_last_cursor_description') or self._last_cursor_description is None:
            return []
            
        return [(col.name, col.type_code) for col in self._last_cursor_description]
    
    def close_connection(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed.") 