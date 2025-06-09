import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, poisson, uniform, chi2
import io # Used to capture print statements

# --- Page Configuration ---
st.set_page_config(
    page_title="🎓 Inventory Management Analysis",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- All Functions from the Original Script ---
# We will place all your functions here. I've made minor adjustments
# to use Streamlit components (e.g., st.write instead of print, st.dataframe instead of display)

# Chapter 1
def load_and_inspect_data(file_path):
    try:
        df = pd.read_csv(file_path)
        st.success(f"✅ Data successfully loaded!")
        st.info("🔍 First 5 rows of the raw data:")
        st.dataframe(df.head())
        st.info("📋 Available columns in the dataset:")
        st.json(df.columns.tolist())
        return df
    except FileNotFoundError:
        st.error(f"❌ Error: The file was not found. Please ensure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during file loading: {e}")
        return None

# Chapter 2
def preprocess_and_aggregate_demand(df, day_col, demand_col):
    if day_col not in df.columns or demand_col not in df.columns:
        st.error(f"❌ Error: One or more required columns ('{day_col}', '{demand_col}') not found.")
        return None

    processed_df = df.copy()
    processed_df[demand_col] = pd.to_numeric(processed_df[demand_col], errors='coerce')
    initial_rows = len(processed_df)
    processed_df.dropna(subset=[demand_col], inplace=True)
    if len(processed_df) < initial_rows:
        st.warning(f"🗑️ Removed {initial_rows - len(processed_df)} rows with non-numeric or missing demand values.")
    else:
        st.success("✅ No non-numeric or missing demand values found to remove.")

    daily_total_demand = processed_df.groupby(day_col)[demand_col].sum().reset_index()
    daily_total_demand.rename(columns={demand_col: 'Total_Demand'}, inplace=True)
    st.info(f"📊 Aggregated total demand for {len(daily_total_demand)} unique days.")
    return daily_total_demand

# Chapter 3
def filter_and_sort_demand(daily_demand_df, max_demand_threshold):
    if daily_demand_df is None or daily_demand_df.empty:
        st.warning("⚠️ No data to filter or sort.")
        return None

    initial_rows = len(daily_demand_df)
    filtered_df = daily_demand_df[daily_demand_df['Total_Demand'] <= max_demand_threshold].copy()
    filtered_rows = len(filtered_df)

    if initial_rows > filtered_rows:
        st.info(f"✂️ Successfully filtered out {initial_rows - filtered_rows} days with Total_Demand > {max_demand_threshold}.")
    else:
        st.success("✅ No demand values found above the threshold to remove.")

    sorted_df = filtered_df.sort_values(by='Total_Demand', ascending=True)
    return sorted_df

# Chapter 4
def calculate_demand_frequency_and_probability(daily_demand_df):
    if daily_demand_df is None or 'Total_Demand' not in daily_demand_df.columns or daily_demand_df.empty:
        st.warning("⚠️ Cannot calculate frequency and probability: 'Total_Demand' column not found or DataFrame is empty.")
        return None
    demand_counts = daily_demand_df['Total_Demand'].value_counts().reset_index()
    demand_counts.columns = ['Total_Demand', 'Frequency']
    total_observations = demand_counts['Frequency'].sum()
    demand_counts['Probability'] = demand_counts['Frequency'] / total_observations
    demand_counts = demand_counts.sort_values(by='Total_Demand').reset_index(drop=True)
    return demand_counts

def analyze_and_visualize_distribution(daily_demand_df, title_suffix=""):
    if daily_demand_df is None or 'Total_Demand' not in daily_demand_df.columns or daily_demand_df.empty:
        return

    st.subheader("📊 Descriptive Statistics")
    st.dataframe(daily_demand_df['Total_Demand'].describe())
    col1, col2 = st.columns(2)
    col1.metric("Skewness", f"{daily_demand_df['Total_Demand'].skew():.4f}")
    col2.metric("Kurtosis", f"{daily_demand_df['Total_Demand'].kurt():.4f}")

    st.subheader("📈 Demand Distribution Histogram")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(daily_demand_df['Total_Demand'], kde=True, bins=30, color='skyblue', edgecolor='black', ax=ax)
    ax.set_title(f'Distribution of Total Daily Demand {title_suffix}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Total Daily Demand (Units)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.grid(axis='y', alpha=0.75, linestyle='--')
    st.pyplot(fig)

# Chapter 5
def calculate_demand_during_lead_time_probability(demand_prob_table, lead_time_days=2):
    if demand_prob_table is None or demand_prob_table.empty:
        return None
    
    lead_time_demand_df = demand_prob_table[['Total_Demand', 'Probability']].copy()
    if lead_time_days == 1:
        lead_time_demand_df.rename(columns={'Total_Demand': 'Demand_During_LeadTime'}, inplace=True)
    else:
        current_df = demand_prob_table[['Total_Demand', 'Probability']].copy()
        for day in range(2, lead_time_days + 1):
            next_day_df = demand_prob_table[['Total_Demand', 'Probability']].copy()
            current_df['_key'] = 1
            next_day_df['_key'] = 1
            combined = pd.merge(current_df, next_day_df, on='_key', suffixes=('_1', '_2'))
            combined['Total_Demand'] = combined['Total_Demand_1'] + combined['Total_Demand_2']
            combined['Probability'] = combined['Probability_1'] * combined['Probability_2']
            current_df = combined[['Total_Demand', 'Probability']]
        lead_time_demand_df = current_df.groupby('Total_Demand')['Probability'].sum().reset_index()
        lead_time_demand_df.rename(columns={'Total_Demand': 'Demand_During_LeadTime'}, inplace=True)
    
    final_lead_time_dist = lead_time_demand_df.sort_values(by='Demand_During_LeadTime').reset_index(drop=True)
    final_lead_time_dist['CSL'] = final_lead_time_dist['Probability'].cumsum()
    return final_lead_time_dist

# Chapter 6
def calculate_expected_shortage(ddlt_prob_table):
    if ddlt_prob_table is None or ddlt_prob_table.empty:
        return None
    df = ddlt_prob_table.copy()
    df = df.sort_values('Demand_During_LeadTime').reset_index(drop=True)
    df['R'] = df['Demand_During_LeadTime']
    df['x_fx'] = df['Demand_During_LeadTime'] * df['Probability']
    
    # Efficient calculation using cumulative sums
    sum_xfx_total = df['x_fx'].sum()
    df['Sum_xfx_Rplus1'] = sum_xfx_total - df['x_fx'].cumsum()
    df['Sum_R_fx_Rplus1'] = df['R'] * (1 - df['CSL'])
    df['E_S'] = (df['Sum_xfx_Rplus1'] - df['Sum_R_fx_Rplus1']).clip(lower=0)
    
    return df[['R', 'Probability', 'CSL', 'E_S']]


# Chapter 8 Functions
def get_es_from_r(R_val, ddlt_table_sorted):
    if R_val in ddlt_table_sorted['R'].values:
        return ddlt_table_sorted.loc[ddlt_table_sorted['R'] == R_val, 'E_S'].iloc[0]
    else:
        idx_gte = ddlt_table_sorted[ddlt_table_sorted['R'] >= R_val].index
        if not idx_gte.empty:
            chosen_R_idx = idx_gte[0]
        else:
            chosen_R_idx = ddlt_table_sorted.index[-1]
        return ddlt_table_sorted.loc[chosen_R_idx, 'E_S']

def calculate_basestock_cost(S_candidate, ddlt_table_sorted, mu_DL, Ch, Cs):
    S_candidate = max(0, S_candidate)
    es_at_S = get_es_from_r(S_candidate, ddlt_table_sorted)
    holding_cost = Ch * (S_candidate - mu_DL + es_at_S) * 365 # Annualized
    shortage_cost = Cs * es_at_S * (D_annual / (S_candidate - mu_DL)) if (S_candidate - mu_DL) > 0 else 0 # Annualized shortage
    total_cost = holding_cost + shortage_cost
    return total_cost

def calculate_s_S_cost(s_candidate, S_candidate, ddlt_table_sorted, D_annual, mu_DL, Cp, Ch, Cs):
    s_candidate = max(0, s_candidate)
    S_candidate = max(s_candidate, S_candidate)
    Q_val = S_candidate - s_candidate
    if Q_val <= 0:
        return float('inf')
    es_at_s = get_es_from_r(s_candidate, ddlt_table_sorted)
    ordering_cost = (D_annual / Q_val) * Cp
    holding_cost = (Q_val / 2 + s_candidate - mu_DL + es_at_s) * Ch * 365 # Annualized
    shortage_cost = (D_annual/ Q_val) * Cs * es_at_s
    total_cost = ordering_cost + holding_cost + shortage_cost
    return total_cost

def grid_search(cost_func, lower_bound, upper_bound, step_size, *cost_args):
    best_x = None
    min_cost = float('inf')
    search_range = np.arange(lower_bound, upper_bound + step_size, step_size)
    
    progress_bar = st.progress(0)
    for i, x in enumerate(search_range):
        current_cost = cost_func(round(x), *cost_args)
        if current_cost < min_cost:
            min_cost = current_cost
            best_x = round(x)
        progress_bar.progress((i + 1) / len(search_range))
    progress_bar.empty()
    return best_x, min_cost

# ... (Cyclic Coordinate Search and other functions would go here if needed)

# --- Streamlit App UI ---
st.title("🎓 Master's Independent Study: Demand & Inventory Analysis")
st.markdown("Welcome! This app guides you through analyzing historical demand data to determine optimal inventory policies.")

# --- Sidebar for Navigation and Inputs ---
st.sidebar.header("⚙️ Control Panel")
uploaded_file = st.sidebar.file_uploader("1. Upload Raw Data CSV", type=['csv'])

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}

if uploaded_file is not None:
    # --- Load and Process Data ---
    if 'raw_data_df' not in st.session_state.processed_data:
         with st.spinner("Loading and inspecting data..."):
            st.session_state.processed_data['raw_data_df'] = load_and_inspect_data(uploaded_file)

    if st.session_state.processed_data.get('raw_data_df') is not None:
        st.sidebar.markdown("---")
        st.sidebar.header("2. Analysis Parameters")
        
        # Chapter 2 & 3 Inputs
        max_demand_threshold = st.sidebar.slider(
            "Demand Outlier Threshold", 
            min_value=300, max_value=1500, value=700, step=10,
            help="Any daily demand exceeding this value will be excluded from the analysis."
        )

        # Chapter 5 Input
        lead_time_days = st.sidebar.slider(
            "Lead Time (Days)",
            min_value=1, max_value=10, value=2, step=1,
            help="The fixed lead time in days for order replenishment."
        )

        # Process data based on sidebar inputs
        raw_df = st.session_state.processed_data['raw_data_df']
        agg_df = preprocess_and_aggregate_demand(raw_df, 'Date', 'Units Sold')
        final_demand_df = filter_and_sort_demand(agg_df, max_demand_threshold)
        st.session_state.processed_data['final_demand_df'] = final_demand_df
        
        # --- Main Panel for Outputs ---
        st.header("Chapter 1-4: Demand Analysis")
        with st.expander("Show/Hide Demand Analysis Details", expanded=True):
            if final_demand_df is not None:
                analyze_and_visualize_distribution(final_demand_df, title_suffix=f"(Max Demand ≤ {max_demand_threshold})")
                
        # --- DDLT and Shortage Calculation ---
        st.header("Chapter 5-6: Lead Time Demand & Expected Shortage")
        with st.expander("Show/Hide Lead Time & Shortage Tables", expanded=False):
            with st.spinner(f"Calculating distribution for a {lead_time_days}-day lead time..."):
                demand_prob_table = calculate_demand_frequency_and_probability(final_demand_df)
                ddlt_prob_table = calculate_demand_during_lead_time_probability(demand_prob_table, lead_time_days)
                st.session_state.processed_data['ddlt_prob_table'] = ddlt_prob_table
            
            if ddlt_prob_table is not in st.session_state.processed_data or st.session_state.processed_data['ddlt_prob_table'] is None:
                 st.error("Could not calculate DDLT probability.")
            else:
                 st.subheader(f"Demand During Lead Time ({lead_time_days} days)")
                 st.dataframe(st.session_state.processed_data['ddlt_prob_table'].head())

                 with st.spinner("Calculating Expected Shortage (E(S))..."):
                    final_ddlt_with_shortage = calculate_expected_shortage(ddlt_prob_table)
                    st.session_state.processed_data['final_ddlt_with_shortage'] = final_ddlt_with_shortage

                 st.subheader("Expected Shortage (E(S)) vs. Reorder Point (R)")
                 st.dataframe(st.session_state.processed_data['final_ddlt_with_shortage'].head())
        
        # --- Chapter 8: Basestock Policy Optimization ---
        st.sidebar.markdown("---")
        st.sidebar.header("3. Cost Parameters")
        
        with st.sidebar.form(key='cost_form'):
            st.subheader("Enter Cost Values")
            cp_cost = st.number_input("Ordering Cost (Cp) / order", min_value=0.0, value=10.0, step=1.0)
            product_cost = st.number_input("Product Cost / unit", min_value=0.0, value=50.0, step=1.0)
            h_percent = st.slider("Daily Holding Cost (h)", 0, 100, 10, help="As a % of product cost per unit-day")
            s_percent = st.slider("Shortage Cost (s)", 0, 100, 30, help="As a % of product cost per unit")
            submitted = st.form_submit_button("🚀 Run Basestock Optimization")

        if submitted:
            if 'final_ddlt_with_shortage' in st.session_state.processed_data and st.session_state.processed_data['final_ddlt_with_shortage'] is not None:
                st.header("Chapter 8: Optimal Basestock Policy")
                
                # Derive parameters
                final_ddlt = st.session_state.processed_data['final_ddlt_with_shortage']
                daily_avg_demand = st.session_state.processed_data['final_demand_df']['Total_Demand'].mean()
                D_annual = daily_avg_demand * 365
                mu_DL = daily_avg_demand * lead_time_days
                Ch = product_cost * (h_percent / 100)
                Cs = product_cost * (s_percent / 100)

                # Define search bounds for Grid Search
                min_R_ddlt = final_ddlt['R'].min()
                max_R_ddlt = final_ddlt['R'].max()
                search_lower_bound = max(0, min_R_ddlt)
                search_upper_bound = max_R_ddlt + 100 # Search a bit beyond the max observed DDLT

                st.info("Performing Grid Search to find the optimal Basestock Level (S)...")
                with st.spinner("Searching... this may take a moment."):
                    optimal_S, min_cost = grid_search(
                        calculate_basestock_cost,
                        search_lower_bound,
                        search_upper_bound,
                        1,  # Step size of 1 unit
                        final_ddlt.sort_values(by='R').reset_index(drop=True),
                        mu_DL, Ch, Cs
                    )

                st.balloons()
                st.success("Optimization Complete!")
                st.subheader("Optimal Basestock (S) System Parameters")

                col1, col2 = st.columns(2)
                col1.metric("Optimal Basestock Level (S)", f"{optimal_S:,.0f} units")
                col2.metric("Minimum Annual Cost", f"{min_cost:,.2f} THB")
                
                with st.expander("View Cost Parameters Used"):
                    st.json({
                        "Ordering Cost (Cp)": f"{cp_cost} THB/order",
                        "Product Cost": f"{product_cost} THB/unit",
                        "Daily Holding Cost (Ch)": f"{Ch:.4f} THB/unit-day",
                        "Shortage Cost (Cs)": f"{Cs:.2f} THB/unit",
                        "Avg Daily Demand": f"{daily_avg_demand:.2f} units/day",
                        "Lead Time": f"{lead_time_days} days"
                    })
            else:
                st.error("Cannot run optimization. Please ensure data is loaded and processed first.")

else:
    st.info("👋 Welcome! Please upload your demand data using the sidebar to begin the analysis.")
