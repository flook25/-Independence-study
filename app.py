# IMPORTANT: This app requires a `requirements.txt` file in the same directory
# with the following content:
# streamlit
# pandas
# matplotlib
# seaborn
# numpy

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="🎓 Inventory Management Analysis",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper function for CSV download ---
@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode('utf-8')

# --- All Functions from the Original Script ---
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
@st.cache_data
def calculate_demand_during_lead_time_probability(demand_prob_table, lead_time_days=2):
    if demand_prob_table is None or demand_prob_table.empty:
        return None
    
    demands = demand_prob_table['Total_Demand'].values
    probs = demand_prob_table['Probability'].values
    
    max_daily_demand = int(demands.max())
    full_probs = np.zeros(max_daily_demand + 1)
    full_probs[demands.astype(int)] = probs

    convolved_probs = full_probs
    for _ in range(lead_time_days - 1):
        convolved_probs = np.convolve(convolved_probs, full_probs)

    final_demands = np.arange(len(convolved_probs))
    final_lead_time_dist = pd.DataFrame({
        'Demand_During_LeadTime': final_demands,
        'Probability': convolved_probs
    })
    
    final_lead_time_dist = final_lead_time_dist[final_lead_time_dist['Probability'] > 1e-9].copy()
    
    final_lead_time_dist = final_lead_time_dist.sort_values(by='Demand_During_LeadTime').reset_index(drop=True)
    final_lead_time_dist['CSL'] = final_lead_time_dist['Probability'].cumsum()
    return final_lead_time_dist

# Chapter 6
@st.cache_data
def calculate_expected_shortage(_ddlt_prob_table): 
    if _ddlt_prob_table is None or _ddlt_prob_table.empty:
        return None
    df = _ddlt_prob_table.copy()
    df = df.sort_values('Demand_During_LeadTime').reset_index(drop=True)
    df['R'] = df['Demand_During_LeadTime']
    df['x_fx'] = df['Demand_During_LeadTime'] * df['Probability']
    
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
        idx = ddlt_table_sorted['R'].searchsorted(R_val, side='left')
        if idx == len(ddlt_table_sorted):
            idx = len(ddlt_table_sorted) - 1
        return ddlt_table_sorted.loc[idx, 'E_S']

def calculate_basestock_cost(S_candidate, ddlt_table_sorted, mu_DL, Ch_annual, Cs, D_annual):
    S_candidate = max(0, S_candidate)
    es_at_S = get_es_from_r(S_candidate, ddlt_table_sorted)
    
    avg_inventory = S_candidate - mu_DL + es_at_S
    holding_cost = Ch_annual * avg_inventory
    shortage_cost = Cs * es_at_S 
    
    total_cost = holding_cost + shortage_cost
    return total_cost

def grid_search(cost_func, lower_bound, upper_bound, step_size, *cost_args):
    best_x = None
    min_cost = float('inf')
    search_range = np.arange(lower_bound, upper_bound + step_size, step_size)
    
    progress_bar = st.progress(0, text="Performing Grid Search...")
    for i, x in enumerate(search_range):
        current_cost = cost_func(round(x), *cost_args)
        if current_cost < min_cost:
            min_cost = current_cost
            best_x = round(x)
        progress_bar.progress((i + 1) / len(search_range))
    progress_bar.empty()
    return best_x, min_cost

# --- Streamlit App UI ---
st.title("🎓 Master's Independent Study: Demand & Inventory Analysis")
st.markdown("#### *Advised by: DR. JIRACHAI BUDDHAKULSOMSIRI*")
st.markdown("---")
st.markdown("Welcome! This app guides you through analyzing historical demand data to determine optimal inventory policies.")

st.sidebar.image("https://admissions.siit.tu.ac.th/wp-content/uploads/2023/06/cropped-TU-SIIT1992-01.png", width=250)
st.sidebar.header("⚙️ Control Panel")
uploaded_file = st.sidebar.file_uploader("1. Upload Raw Data CSV", type=['csv'])

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}

if uploaded_file is not None:
    if 'raw_data_df' not in st.session_state.processed_data:
         with st.spinner("Loading and inspecting data..."):
            st.session_state.processed_data['raw_data_df'] = load_and_inspect_data(uploaded_file)

    if st.session_state.processed_data.get('raw_data_df') is not None:
        st.sidebar.markdown("---")
        st.sidebar.header("2. Analysis Parameters")
        
        max_demand_threshold = st.sidebar.slider(
            "Demand Outlier Threshold", min_value=300, max_value=1500, value=700, step=10,
            help="Any daily demand exceeding this value will be excluded from the analysis."
        )
        lead_time_days = st.sidebar.slider(
            "Lead Time (Days)", min_value=1, max_value=10, value=2, step=1,
            help="The fixed lead time in days for order replenishment."
        )

        raw_df = st.session_state.processed_data['raw_data_df']
        agg_df = preprocess_and_aggregate_demand(raw_df, 'Date', 'Units Sold')
        final_demand_df = filter_and_sort_demand(agg_df, max_demand_threshold)
        st.session_state.processed_data['final_demand_df'] = final_demand_df
        
        st.header("Chapter 1-4: Demand Analysis")
        with st.expander("Show/Hide Demand Analysis Details", expanded=True):
            if final_demand_df is not None:
                analyze_and_visualize_distribution(final_demand_df, title_suffix=f"(Max Demand ≤ {max_demand_threshold})")
                
        st.header("Chapter 5-6: Lead Time Demand & Expected Shortage")
        with st.expander("Show/Hide Lead Time & Shortage Tables", expanded=True): # Changed to expanded=True for visibility
            with st.spinner(f"Calculating distribution for a {lead_time_days}-day lead time..."):
                demand_prob_table = calculate_demand_frequency_and_probability(final_demand_df)
                ddlt_prob_table = calculate_demand_during_lead_time_probability(demand_prob_table, lead_time_days)
            
            if ddlt_prob_table is None:
                st.error("Could not calculate DDLT probability.")
            else:
                st.session_state.processed_data['ddlt_prob_table'] = ddlt_prob_table
                st.subheader(f"Demand During Lead Time ({lead_time_days} days)")
                
                # --- CHANGE 1: Display full DataFrame ---
                st.dataframe(ddlt_prob_table)
                
                # --- CHANGE 2: Add Download Button ---
                csv_ddlt = convert_df_to_csv(ddlt_prob_table)
                st.download_button(
                    label="📥 Download DDLT Data as CSV",
                    data=csv_ddlt,
                    file_name=f'ddlt_probability_{lead_time_days}days.csv',
                    mime='text/csv',
                )

                with st.spinner("Calculating Expected Shortage (E(S))..."):
                    final_ddlt_with_shortage = calculate_expected_shortage(ddlt_prob_table)
                    st.session_state.processed_data['final_ddlt_with_shortage'] = final_ddlt_with_shortage

                st.subheader("Expected Shortage (E(S)) vs. Reorder Point (R)")
                
                # --- CHANGE 1: Display full DataFrame ---
                st.dataframe(final_ddlt_with_shortage)

                # --- CHANGE 2: Add Download Button ---
                csv_es = convert_df_to_csv(final_ddlt_with_shortage)
                st.download_button(
                    label="📥 Download E(S) Data as CSV",
                    data=csv_es,
                    file_name=f'expected_shortage_{lead_time_days}days.csv',
                    mime='text/csv',
                )
        
        st.sidebar.markdown("---")
        st.sidebar.header("3. Cost Parameters")
        
        with st.sidebar.form(key='cost_form'):
            st.subheader("Enter Cost Values")
            cp_cost = st.number_input("Ordering Cost (Cp) / order", min_value=0.0, value=10.0, step=1.0)
            product_cost = st.number_input("Product Cost / unit", min_value=0.0, value=50.0, step=1.0)
            h_percent_annual = st.slider("Annual Holding Rate (h)", 0, 100, 10, help="As an annual % of product cost.")
            s_percent = st.slider("Shortage Cost Rate (s)", 0, 100, 30, help="As a % of product cost per unit.")
            submitted = st.form_submit_button("🚀 Run Basestock Optimization")

        if submitted:
            if st.session_state.processed_data.get('final_ddlt_with_shortage') is not None:
                st.header("Chapter 8: Optimal Basestock Policy")
                
                final_ddlt = st.session_state.processed_data['final_ddlt_with_shortage']
                daily_avg_demand = st.session_state.processed_data['final_demand_df']['Total_Demand'].mean()
                D_annual = daily_avg_demand * 365
                mu_DL = daily_avg_demand * lead_time_days
                Ch_annual = product_cost * (h_percent_annual / 100)
                Cs = product_cost * (s_percent / 100)

                min_R_ddlt = final_ddlt['R'].min()
                max_R_ddlt = final_ddlt['R'].max()
                search_lower_bound = max(0, int(min_R_ddlt))
                search_upper_bound = int(max_R_ddlt + (daily_avg_demand * 5))

                with st.spinner("Searching for optimal Basestock Level (S)... This may take a moment."):
                    optimal_S, min_cost = grid_search(
                        calculate_basestock_cost,
                        search_lower_bound,
                        search_upper_bound,
                        1,
                        final_ddlt.sort_values(by='R').reset_index(drop=True),
                        mu_DL, Ch_annual, Cs, D_annual
                    )

                st.balloons()
                st.success("Optimization Complete!")
                st.subheader("Optimal Basestock (S) System Parameters")

                col1, col2 = st.columns(2)
                col1.metric("Optimal Basestock Level (S)", f"{optimal_S:,.0f} units")
                col2.metric("Minimum Annual Cost", f"{min_cost:,.2f} THB")
                
                with st.expander("View Cost & Demand Parameters Used"):
                    st.json({
                        "Ordering Cost (Cp)": f"{cp_cost} THB/order (Note: Not used in Basestock cost formula)",
                        "Product Cost": f"{product_cost} THB/unit",
                        "Annual Holding Cost (Ch)": f"{Ch_annual:.4f} THB/unit-year",
                        "Shortage Cost (Cs)": f"{Cs:.2f} THB/unit",
                        "Avg Daily Demand": f"{daily_avg_demand:.2f} units/day",
                        "Annual Demand (D)": f"{D_annual:,.0f} units/year",
                        "Avg Demand During Lead Time (μ_DL)": f"{mu_DL:,.2f} units",
                        "Lead Time": f"{lead_time_days} days"
                    })
            else:
                st.error("Cannot run optimization. Please ensure data is loaded and processed first.")
else:
    st.info("👋 Welcome! Please upload your demand data using the sidebar to begin the analysis.")
