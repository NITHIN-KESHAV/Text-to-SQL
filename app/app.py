import streamlit as st
import pandas as pd
import plotly.express as px
import time
import sys
import os
import base64

# Import agent from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from postgresql_agent.my_agent.LangGraphAgent import LangGraphAgent

# Page configuration
st.set_page_config(
    page_title="Text to SQL Application",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# IMDB logo component
def add_logo():
    imdb_logo_html = '''
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg" alt="IMDB Logo" width="120" />
    </div>
    '''
    return imdb_logo_html

# IMDB theme styling
st.markdown("""
<style>
    /* Global color scheme */
    :root {
        --imdb-yellow: #F5C518;
        --imdb-black: #000000;
        --imdb-dark-gray: #1F1F1F;
        --imdb-light-gray: #F5F5F5;
    }
    
    /* Background color */
    .stApp {
        background-color: #1F1F1F;
        color: white !important;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
        color: var(--imdb-yellow) !important;
    }
    
    .sub-header {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 500;
        color: var(--imdb-yellow) !important;
    }
    
    /* Info box */
    .info-box {
        background-color: var(--imdb-dark-gray);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid var(--imdb-yellow);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: var(--imdb-yellow) !important;
        color: var(--imdb-black) !important;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none !important;
    }
    
    .stButton>button:hover {
        background-color: #D9AD16 !important;
    }
    
    /* Text inputs */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 0.5rem;
        background-color: #333333;
        color: white;
        border: 1px solid var(--imdb-yellow);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        color: white !important;
        background-color: #333333;
    }
    
    div.stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }

    div.stTabs [data-baseweb="tab"] {
        background-color: #333333;
        border-radius: 4px 4px 0 0;
        color: white;
        padding: 8px 16px;
        font-size: 16px;
        font-weight: 500;
    }

    div.stTabs [aria-selected="true"] {
        background-color: var(--imdb-yellow);
        color: var(--imdb-black);
    }
    
    /* Links */
    a {
        color: var(--imdb-yellow) !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #121212;
    }
    
    /* Footer */
    footer {
        color: white !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        border: 1px solid #333333;
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Display logo
st.markdown(add_logo(), unsafe_allow_html=True)

# Initialize state
if "agent" not in st.session_state:
    st.session_state.agent = LangGraphAgent(debug=False)

if "history" not in st.session_state:
    st.session_state.history = []

if "generated_sql" not in st.session_state:
    st.session_state.generated_sql = ""

# Sidebar content
with st.sidebar:
    st.title("Database Information")
    
    st.markdown("""
    ### PostgreSQL Pagila Database
    
    This application uses the Pagila database, a sample database for PostgreSQL that models a 
    DVD rental store. It includes tables for:
    
    - **Films**: Movie information
    - **Actors**: Actor information and film appearances
    - **Categories**: Film genres
    - **Customers**: Customer data
    - **Rentals**: Rental history
    - **Payments**: Payment information
    """)
    
    with st.expander("Full Database Schema"):
        st.markdown("""
        ### Core Tables
        
        **film**
        - `film_id` (PK): Unique identifier
        - `title`: Film title
        - `description`: Plot summary
        - `release_year`: Year of release
        - `language_id` (FK): Reference to language
        - `rental_duration`: Standard rental period
        - `rental_rate`: Standard rental fee
        - `length`: Length in minutes
        - `replacement_cost`: Cost to replace if damaged
        - `rating`: Content rating (G, PG, etc.)
        
        **actor**
        - `actor_id` (PK): Unique identifier
        - `first_name`: Actor's first name
        - `last_name`: Actor's last name
        
        **category**
        - `category_id` (PK): Unique identifier 
        - `name`: Category name (Action, Comedy, etc.)
        
        **customer**
        - `customer_id` (PK): Unique identifier
        - `store_id` (FK): Associated store
        - `first_name`: Customer's first name
        - `last_name`: Customer's last name
        - `email`: Customer's email address
        - `address_id` (FK): Reference to address
        - `active`: Whether customer account is active
        
        ### Junction Tables
        
        **film_actor**
        - `actor_id` (PK/FK): Reference to actor
        - `film_id` (PK/FK): Reference to film
        
        **film_category**
        - `film_id` (PK/FK): Reference to film
        - `category_id` (PK/FK): Reference to category
        
        ### Transaction Tables
        
        **rental**
        - `rental_id` (PK): Unique identifier
        - `rental_date`: Date/time item rented
        - `inventory_id` (FK): Reference to inventory item
        - `customer_id` (FK): Reference to customer
        - `return_date`: Date/time returned (null if not returned)
        - `staff_id` (FK): Staff member who processed
        
        **payment**
        - `payment_id` (PK): Unique identifier
        - `customer_id` (FK): Reference to customer
        - `staff_id` (FK): Staff who processed payment
        - `rental_id` (FK): Associated rental
        - `amount`: Payment amount
        - `payment_date`: Date/time of payment
        
        ### Inventory/Store Tables
        
        **inventory**
        - `inventory_id` (PK): Unique identifier
        - `film_id` (FK): Reference to film
        - `store_id` (FK): Store where item is held
        
        **store**
        - `store_id` (PK): Unique identifier
        - `manager_staff_id` (FK): Reference to staff
        - `address_id` (FK): Store address
        
        ### Support Tables
        
        **language**
        - `language_id` (PK): Unique identifier
        - `name`: Language name
        
        **staff**
        - `staff_id` (PK): Unique identifier
        - `first_name`: Staff first name
        - `last_name`: Staff last name
        - `address_id` (FK): Reference to address
        - `email`: Staff email address
        - `store_id` (FK): Associated store
        - `active`: Whether staff member is active
        - `username`: Login username
        - `password`: Login password
        
        **address**
        - `address_id` (PK): Unique identifier
        - `address`: Street address
        - `district`: District/state/province
        - `city_id` (FK): Reference to city
        - `postal_code`: Zip/postal code
        - `phone`: Phone number
        
        **city**
        - `city_id` (PK): Unique identifier
        - `city`: City name
        - `country_id` (FK): Reference to country
        
        **country**
        - `country_id` (PK): Unique identifier
        - `country`: Country name
        """)
    
    # Schema overview
    st.markdown("""
    ### Schema Overview
    
    Key tables:
    - `film`: Main film information
    - `actor`: Actor information
    - `category`: Film categories/genres
    - `film_actor`: Maps actors to films
    - `film_category`: Maps films to categories
    - `customer`: Customer data
    - `rental`: Rental transactions
    - `payment`: Payment records
    """)
    
    with st.expander("Architecture Details"):
        st.markdown("""
        This application uses a LangGraph-based agent that:
        
        1. Analyzes the database schema
        2. Converts natural language to SQL
        3. Validates and fixes SQL if needed
        4. Executes the query
        5. Returns results with visualization
        
        The agent follows a directed graph workflow with nodes for 
        each processing step and uses an LLM to understand natural 
        language and generate appropriate SQL.
        """)
    
    # Query history
    st.subheader("Query History")
    if st.session_state.history:
        for i, (query, _) in enumerate(st.session_state.history):
            if st.button(f"{i+1}. {query[:40]}...", key=f"history_{i}"):
                st.session_state.user_query = query

# Main content
st.markdown('<p class="main-header">Text to SQL LLM Application</p>', unsafe_allow_html=True)

st.markdown("""
This application allows you to query a PostgreSQL database using natural language.
Simply type your question, and our LLM agent will convert it to SQL and fetch the results.
""")

# DB connection info
with st.expander("Database Connection Details"):
    st.code("""
# PostgreSQL Connection
Host: localhost
Port: 5432
Database: pagila
Schema: public
    """)

# User input
user_input = st.text_area(
    "Enter your question about the movie rental database:",
    value=st.session_state.get("user_query", ""),
    height=100,
    key="user_query",
    placeholder="Example: Show me the top 5 most rented films"
)

# Execute query button
col1, col2 = st.columns([4, 1])
with col2:
    execute_button = st.button("Execute Query", type="primary", use_container_width=True)

# Query processing
if execute_button:
    if user_input:
        with st.spinner("Processing your query..."):
            # Start time tracking
            start_time = time.time()
            
            # Process query
            result = st.session_state.agent.process_query(user_input)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update history
            st.session_state.history.insert(0, (user_input, result))
            
            # Store SQL
            st.session_state.generated_sql = result.get("generated_sql", "")
            
            # Results header
            st.markdown('<p class="sub-header">Results</p>', unsafe_allow_html=True)
            
            # Processing info
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Query processed in {processing_time:.2f} seconds")
            with col2:
                if result.get("error"):
                    st.error("Query Error")
                else:
                    st.success("Query Successful")
            
            # SQL display
            if result.get("generated_sql"):
                with st.expander("Generated SQL", expanded=True):
                    st.code(result["generated_sql"], language="sql")
            
            # Error display
            if result.get("error"):
                st.error(f"Error: {result['error']}")
            
            # Results display
            if result.get("results"):
                # Create dataframe
                if result.get("column_types"):
                    columns = [col_name for col_name, _ in result["column_types"]]
                    df = pd.DataFrame(result["results"], columns=columns)
                else:
                    df = pd.DataFrame(result["results"])
                
                # Table view
                st.dataframe(df, use_container_width=True)
                
                # Visualization
                if len(df) > 0:
                    st.markdown('<p class="sub-header">Visualization</p>', unsafe_allow_html=True)
                    
                    # Tabs
                    tab1, tab2, tab3 = st.tabs(["Table", "Chart", "Statistics"])
                    
                    with tab1:
                        # Table view
                        st.dataframe(df, use_container_width=True)
                    
                    with tab2:
                        # Chart view
                        if len(df.columns) >= 2:
                            # Find numeric columns
                            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                            
                            if numeric_cols:
                                # Set y-axis (numeric)
                                y_col = numeric_cols[0]
                                # Set x-axis (non-numeric preferred)
                                non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
                                x_col = non_numeric_cols[0] if non_numeric_cols else df.columns[0]
                                
                                st.subheader("Data Visualization")
                                
                                # Chart selector
                                chart_type = st.selectbox(
                                    "Select chart type:",
                                    ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart"],
                                    key="chart_type"
                                )
                                
                                if chart_type == "Bar Chart":
                                    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                elif chart_type == "Line Chart":
                                    fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                elif chart_type == "Scatter Plot":
                                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                elif chart_type == "Pie Chart" and len(df) <= 20:
                                    fig = px.pie(df, names=x_col, values=y_col, title=f"Distribution of {y_col}")
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No numeric columns found for visualization. Charts require at least one numeric column.")
                        else:
                            st.info("Need at least two columns to create a visualization.")
                    
                    with tab3:
                        # Statistics view
                        st.subheader("Data Statistics")
                        
                        # Summary statistics
                        st.write("Dataset Shape:", df.shape)
                        st.write("Total Records:", len(df))
                        
                        # Numeric statistics
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 0:
                            st.write("Numeric Columns Statistics:")
                            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                        else:
                            st.info("No numeric columns found for statistics")
                        
                        # Column types
                        st.write("Column Data Types:")
                        dtypes_df = pd.DataFrame(
                            {"Column": df.columns, "Data Type": [str(df[col].dtype) for col in df.columns]}
                        )
                        st.dataframe(dtypes_df, use_container_width=True)
    else:
        st.warning("Please enter a question to query the database.")

# Footer
st.markdown("---")
st.markdown(
    "**Text to SQL LLM Application** powered by LangGraph and PostgreSQL."
) 