# IMPORTANT: This app requires a `requirements.txt` file in the same directory
# with the following content:
# streamlit
# pandas
# matplotlib
# seaborn
# numpy
# openpyxl
# plotly

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# --- Page Configuration ---
st.set_page_config(
    page_title="🎓 Advanced Inventory Management Analysis",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2980b9);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2980b9;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stProgress .st-bo {
        background-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🎓 Advanced Inventory Management Analysis</h1>
    <h3>Master's Independent Study</h3>
    <p><strong>Advised by:</strong> DR. JIRACHAI BUDDHAKULSOMSIRI</p>
</div>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode('utf-8')

def create_progress_indicator(step, total_steps=8):
    """Creates a visual progress indicator"""
    progress = step / total_steps
    st.progress(progress)
    st.caption(f"Step {step} of {total_steps} completed")

def display_metric_card(title, value, delta=None, help_text=None):
    """Creates a styled metric card"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.metric(title, value, delta=delta, help=help_text)

# --- All Original Functions (Enhanced with Progress Tracking) ---
def load_and_inspect_data(uploaded_file):
    """Enhanced data loading with progress tracking and validation"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Reading file...")
        progress_bar.progress(25)
        
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Unsupported file format. Please upload a CSV or XLSX file.")
            return None
            
        progress_bar.progress(50)
        status_text.text("🔍 Validating data structure...")
        
        # Enhanced data validation
        if df.empty:
            st.error("❌ The uploaded file is empty.")
            return None
            
        progress_bar.progress(75)
        status_text.text("✅ Processing complete!")
        progress_bar.progress(100)
        
        # Enhanced display with tabs
        tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📋 Data Info", "🔍 Column Analysis"])
        
        with tab1:
            st.success(f"✅ Data successfully loaded: **{len(df):,}** rows × **{len(df.columns)}** columns")
            st.dataframe(df.head(10), use_container_width=True)
            
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.info("**Dataset Shape**")
                st.write(f"Rows: {len(df):,}")
                st.write(f"Columns: {len(df.columns)}")
                st.write(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            with col2:
                st.info("**Data Types**")
                dtype_counts = df.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    st.write(f"{dtype}: {count}")
        
        with tab3:
            st.info("**Available Columns**")
            for i, col in enumerate(df.columns, 1):
                st.write(f"{i}. **{col}** ({df[col].dtype})")
                
        status_text.empty()
        progress_bar.empty()
        return df
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error loading file: {str(e)}")
        return None

def preprocess_and_aggregate_demand(df, day_col, demand_col):
    """Enhanced preprocessing with detailed feedback"""
    if day_col not in df.columns or demand_col not in df.columns:
        st.error(f"❌ Required columns not found: '{day_col}', '{demand_col}'")
        available_cols = ", ".join(df.columns.tolist())
        st.info(f"Available columns: {available_cols}")
        return None
    
    with st.spinner("Processing demand data..."):
        processed_df = df.copy()
        
        # Convert demand to numeric
        processed_df[demand_col] = pd.to_numeric(processed_df[demand_col], errors='coerce')
        
        # Track data quality
        initial_rows = len(processed_df)
        null_rows = processed_df[demand_col].isnull().sum()
        negative_rows = len(processed_df[processed_df[demand_col] < 0])
        
        # Clean data
        processed_df = processed_df.dropna(subset=[demand_col])
        processed_df = processed_df[processed_df[demand_col] >= 0]
        
        # Aggregate by day
        daily_total_demand = processed_df.groupby(day_col)[demand_col].sum().reset_index()
        daily_total_demand.rename(columns={demand_col: 'Total_Demand'}, inplace=True)
        
        # Display cleaning results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Original Rows", f"{initial_rows:,}")
        with col2:
            st.metric("Null Values Removed", f"{null_rows:,}", delta=f"-{null_rows}")
        with col3:
            st.metric("Negative Values Removed", f"{negative_rows:,}", delta=f"-{negative_rows}")
        with col4:
            st.metric("Unique Days", f"{len(daily_total_demand):,}")
            
        if null_rows > 0 or negative_rows > 0:
            st.warning(f"🧹 Data cleaned: Removed {null_rows + negative_rows} problematic rows")
        else:
            st.success("✅ No data cleaning required - all values are valid!")
            
        return daily_total_demand

def filter_and_sort_demand(daily_demand_df, max_demand_threshold):
    """Enhanced filtering with visual feedback"""
    if daily_demand_df is None or daily_demand_df.empty:
        st.warning("⚠️ No data available for filtering")
        return None
    
    initial_rows = len(daily_demand_df)
    outliers = len(daily_demand_df[daily_demand_df['Total_Demand'] > max_demand_threshold])
    
    filtered_df = daily_demand_df[daily_demand_df['Total_Demand'] <= max_demand_threshold].copy()
    sorted_df = filtered_df.sort_values(by='Total_Demand', ascending=True)
    
    # Visual feedback
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Days Before Filter", f"{initial_rows:,}")
    with col2:
        st.metric("Outliers Removed", f"{outliers:,}", delta=f"-{outliers}")
    with col3:
        st.metric("Days After Filter", f"{len(sorted_df):,}")
    
    if outliers > 0:
        st.info(f"✂️ Filtered out {outliers} days with demand > {max_demand_threshold:,} units")
    else:
        st.success("✅ No outliers found - all data retained!")
        
    return sorted_df

def calculate_demand_frequency_and_probability(daily_demand_df):
    """Enhanced probability calculation with validation"""
    if daily_demand_df is None or 'Total_Demand' not in daily_demand_df.columns or daily_demand_df.empty:
        st.warning("⚠️ Cannot calculate probabilities - invalid data")
        return None
    
    demand_counts = daily_demand_df['Total_Demand'].value_counts().reset_index()
    demand_counts.columns = ['Total_Demand', 'Frequency']
    demand_counts['Probability'] = demand_counts['Frequency'] / demand_counts['Frequency'].sum()
    demand_counts = demand_counts.sort_values(by='Total_Demand').reset_index(drop=True)
    
    # Validation check
    prob_sum = demand_counts['Probability'].sum()
    if abs(prob_sum - 1.0) > 0.001:
        st.warning(f"⚠️ Probability sum validation: {prob_sum:.6f} (should be 1.0)")
    else:
        st.success("✅ Probability distribution validated")
    
    return demand_counts

def analyze_and_visualize_distribution(daily_demand_df, title_suffix=""):
    """Enhanced visualization with interactive plots"""
    if daily_demand_df is None or 'Total_Demand' not in daily_demand_df.columns or daily_demand_df.empty:
        return
    
    demand_data = daily_demand_df['Total_Demand']
    
    # Statistical Summary and Metrics
    st.subheader("📊 Descriptive Statistics")
    st.dataframe(demand_data.describe(), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Skewness", f"{demand_data.skew():.4f}")
    with col2:
        st.metric("Kurtosis", f"{demand_data.kurt():.4f}")
    
    # Interactive visualizations
    st.subheader("📈 Interactive Demand Analysis")
    
    tab1, tab2 = st.tabs(["📊 Distribution", "📈 Time Series"])
    
    with tab1:
        # Modernized Histogram with dark background
        fig_hist, ax_hist = plt.subplots(figsize=(12, 7))
        fig_hist.patch.set_facecolor('#222c36')
        ax_hist.set_facecolor('#222c36')
        # Histogram
        sns.histplot(
            demand_data, kde=False, bins=30, color='#4fc3f7', edgecolor='#90caf9',
            alpha=0.7, ax=ax_hist
        )
        # KDE
        sns.kdeplot(
            demand_data, color='#ffb300', linewidth=3, ax=ax_hist
        )
        # Titles and labels
        ax_hist.set_title(f'Distribution of Total Daily Demand {title_suffix}', fontsize=18, fontweight='bold', color='white')
        ax_hist.set_xlabel('Total Daily Demand (Units)', fontsize=15, color='white')
        ax_hist.set_ylabel('Frequency', fontsize=15, color='white')
        # Ticks and grid
        ax_hist.tick_params(axis='x', colors='white')
        ax_hist.tick_params(axis='y', colors='white')
        ax_hist.grid(axis='y', alpha=0.3, linestyle='--', color='white')
        for spine in ax_hist.spines.values():
            spine.set_color('white')
        st.pyplot(fig_hist)
    
    with tab2:
        if 'Date' in daily_demand_df.columns:
            df_sorted = daily_demand_df.sort_values('Date').copy()
            if len(df_sorted) >= 7:
                df_sorted['RollingMean'] = df_sorted['Total_Demand'].rolling(window=7, min_periods=1).mean()
            else:
                df_sorted['RollingMean'] = df_sorted['Total_Demand']
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=df_sorted['Date'],
                y=df_sorted['Total_Demand'],
                mode='markers',
                marker=dict(color='rgba(41,128,185,0.5)', size=6),
                name='Daily Demand',
                hovertemplate='Date=%{x}<br>Daily Demand=%{y}'
            ))
            fig_ts.add_trace(go.Scatter(
                x=df_sorted['Date'],
                y=df_sorted['RollingMean'],
                mode='lines',
                line=dict(color='#e67e22', width=2),
                name='7-Day Rolling Mean',
                hovertemplate='Date=%{x}<br>7-Day Mean=%{y:.2f}'
            ))
            fig_ts.update_layout(
                title='Demand Over Time',
                xaxis_title='Date',
                yaxis_title='Daily Demand (Units)',
                plot_bgcolor='#222c36',
                paper_bgcolor='#222c36',
                font_color='white',
                xaxis=dict(showgrid=True, gridcolor='gray', tickformat='%b %Y', tickangle=45, color='white'),
                yaxis=dict(showgrid=True, gridcolor='gray', color='white'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("Date column not available for time series analysis")

@st.cache_data
def calculate_demand_during_lead_time_probability(demand_prob_table, lead_time_days=2):
    """Enhanced DDLT calculation with progress tracking"""
    if demand_prob_table is None or demand_prob_table.empty:
        return None
    
    with st.spinner(f"Computing demand distribution for {lead_time_days}-day lead time..."):
        demands = demand_prob_table['Total_Demand'].values
        probs = demand_prob_table['Probability'].values
        max_daily_demand = int(demands.max())
        
        # Create full probability array
        full_probs = np.zeros(max_daily_demand + 1)
        full_probs[demands.astype(int)] = probs
        
        # Convolve for lead time days
        convolved_probs = full_probs
        for day in range(lead_time_days - 1):
            progress = (day + 1) / (lead_time_days - 1)
            convolved_probs = np.convolve(convolved_probs, full_probs)
        
        # Create final distribution
        final_demands = np.arange(len(convolved_probs))
        final_lead_time_dist = pd.DataFrame({
            'Demand_During_LeadTime': final_demands, 
            'Probability': convolved_probs
        })
        
        # Filter out negligible probabilities
        final_lead_time_dist = final_lead_time_dist[final_lead_time_dist['Probability'] > 1e-9].copy()
        final_lead_time_dist['CSL'] = final_lead_time_dist['Probability'].cumsum()
        
        return final_lead_time_dist.sort_values(by='Demand_During_LeadTime').reset_index(drop=True)

@st.cache_data
def calculate_expected_shortage(_ddlt_prob_table):
    """Enhanced expected shortage calculation"""
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

def find_optimal_qr(ddlt_with_shortage_table, daily_avg_demand, avg_lead_time,
                    cp_ordering_cost, product_cost, h_percent_annual, s_percent,
                    service_level_type='backlog', convergence_tolerance=0.001, max_iterations=100):
    """Enhanced Q,R optimization with detailed progress tracking"""
    
    if ddlt_with_shortage_table is None or ddlt_with_shortage_table.empty:
        st.error("⚠️ Invalid DDLT table for optimization")
        return None

    # Initialize parameters
    days_per_year = 365
    Cp = cp_ordering_cost
    D_annual = daily_avg_demand * days_per_year
    mu_DL = daily_avg_demand * avg_lead_time
    Ch_annual = product_cost * h_percent_annual
    Cs_per_unit = product_cost * s_percent

    # Initial Q (EOQ)
    Q = np.sqrt((2 * D_annual * Cp) / Ch_annual)

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    Q_old = 0.0
    iteration = 0
    optimization_history = []
    ddlt_table_sorted = ddlt_with_shortage_table.sort_values(by='R').reset_index(drop=True)

    while abs(Q - Q_old) > convergence_tolerance and iteration < max_iterations:
        Q_old = Q
        iteration += 1
        
        # Update progress
        progress = min(iteration / max_iterations, 0.95)
        progress_bar.progress(progress)
        status_text.text(f"Iteration {iteration}: Optimizing Q={Q:.1f}, convergence check...")

        # Determine optimal CSL
        if service_level_type == 'backlog':
            csl_optimal = 1 - (Q * Ch_annual) / (D_annual * Cs_per_unit)
        elif service_level_type == 'lost_sales':
            csl_optimal = (D_annual * Cs_per_unit) / ((D_annual * Cs_per_unit) + (Q * Ch_annual))
        else:
            st.error(f"❌ Invalid service level type: '{service_level_type}'")
            return None
            
        csl_optimal = max(0, min(1, csl_optimal))

        # Find reorder point
        r_candidates = ddlt_table_sorted[ddlt_table_sorted['CSL'] >= csl_optimal]
        if not r_candidates.empty:
            R = r_candidates.iloc[0]['R']
        else:
            R = ddlt_table_sorted['R'].max()

        # Calculate costs
        es_at_R_row = ddlt_table_sorted.loc[ddlt_table_sorted['R'] == R, 'E_S']
        es_at_R = es_at_R_row.iloc[0] if not es_at_R_row.empty else 0.0

        ordering_cost = (D_annual / Q) * Cp
        if service_level_type == 'backlog':
            holding_cost = ((Q / 2) + R - mu_DL) * Ch_annual
        else:  # Lost Sales
            holding_cost = ((Q / 2) + R - mu_DL + es_at_R) * Ch_annual
        shortage_cost = (D_annual / Q) * Cs_per_unit * es_at_R
        TAC = ordering_cost + holding_cost + shortage_cost

        # Log history
        optimization_history.append({
            'Iteration': iteration,
            'Q_old': Q_old,
            'CSL_optimal': csl_optimal,
            'R_found': R,
            'E_S_at_R': es_at_R,
            'Ordering_Cost': ordering_cost,
            'Holding_Cost': holding_cost,
            'Shortage_Cost': shortage_cost,
            'TAC': TAC
        })

        # Recompute Q
        Q = np.sqrt((2 * D_annual * (Cp + Cs_per_unit * es_at_R)) / Ch_annual)

    # Finalize
    progress_bar.progress(1.0)
    status_text.text("✅ Optimization complete!")
    
    final_q = Q
    final_r = R
    final_tac = TAC

    if iteration >= max_iterations:
        st.warning(f"⚠️ Reached maximum iterations ({max_iterations}). Solution may not be fully converged.")
    else:
        st.success(f"🎉 Converged in {iteration} iterations!")

    progress_bar.empty()
    status_text.empty()

    return {
        'optimal_Q': final_q,
        'optimal_R': final_r,
        'min_TAC': final_tac,
        'convergence_iterations': iteration,
        'history': pd.DataFrame(optimization_history)
    }

# --- Enhanced Streamlit App UI ---
def main():
    # Sidebar with enhanced navigation
    st.sidebar.image("https://admissions.siit.tu.ac.th/wp-content/uploads/2023/06/cropped-TU-SIIT1992-01.png", width=250)
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Navigation")
    
    
    # File upload section
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Demand Data", 
        type=['csv', 'xlsx'],
        help="Upload your historical demand data in CSV or Excel format"
    )
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {}

    if uploaded_file is not None:
        # Data loading and processing
        if ('file_name' not in st.session_state.processed_data or 
            st.session_state.processed_data['file_name'] != uploaded_file.name):
            
            st.header("🔄 Step 1: Data Loading & Inspection")
            create_progress_indicator(1)
            
            st.session_state.processed_data['raw_data_df'] = load_and_inspect_data(uploaded_file)
            st.session_state.processed_data['file_name'] = uploaded_file.name

        if st.session_state.processed_data.get('raw_data_df') is not None:
            raw_df = st.session_state.processed_data['raw_data_df']
            
            # Analysis parameters
            st.sidebar.markdown("---")
            st.sidebar.header("⚙️ Analysis Parameters")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                max_demand_threshold = st.slider(
                    "Max Daily Demand", 
                    int(raw_df['Units Sold'].min()) if 'Units Sold' in raw_df.columns else 300,
                    int(raw_df['Units Sold'].max() * 2.0) if 'Units Sold' in raw_df.columns else 10000, # Increased upper bound to 10000
                    700, 10,
                    help="Filter out extreme demand values"
                )
            with col2:
                lead_time_days = st.slider(
                    "Lead Time (Days)", 
                    1, 20, 2, 1, # Increased upper bound
                    help="Supply lead time in days"
                )

            # Data preprocessing
            st.header("📊 Step 2-3: Data Processing & Filtering")
            create_progress_indicator(3)
            
            with st.expander("🔍 View Processing Details", expanded=False):
                agg_df = preprocess_and_aggregate_demand(raw_df, 'Date', 'Units Sold')
                final_demand_df = filter_and_sort_demand(agg_df, max_demand_threshold)
                st.session_state.processed_data['final_demand_df'] = final_demand_df

            # Demand analysis
            st.header("📈 Step 4: Demand Distribution Analysis")
            create_progress_indicator(4)
            
            final_demand_df = st.session_state.processed_data['final_demand_df']
            analyze_and_visualize_distribution(
                final_demand_df, 
                title_suffix=f"(Filtered ≤ {max_demand_threshold:,})"
            )

            # DDLT and shortage calculations
            st.header("⏱️ Step 5-6: Lead Time Analysis & Expected Shortage")
            create_progress_indicator(6)
            
            with st.expander("📋 View Lead Time Distribution Tables", expanded=False):
                demand_prob_table = calculate_demand_frequency_and_probability(final_demand_df)
                ddlt_prob_table = calculate_demand_during_lead_time_probability(
                    demand_prob_table, lead_time_days
                )

                if ddlt_prob_table is not None:
                    st.session_state.processed_data['ddlt_prob_table'] = ddlt_prob_table
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.subheader(f"Lead Time Demand ({lead_time_days} days)")
                        st.dataframe(ddlt_prob_table, use_container_width=True)
                        
                    final_ddlt_with_shortage = calculate_expected_shortage(ddlt_prob_table)
                    st.session_state.processed_data['final_ddlt_with_shortage'] = final_ddlt_with_shortage
                    
                    with col2:
                        st.subheader("Expected Shortage Analysis")
                        st.dataframe(final_ddlt_with_shortage, use_container_width=True)
                        
                    # Download buttons
                    st.download_button(
                        label="📥 Download DDLT Table",
                        data=convert_df_to_csv(ddlt_prob_table),
                        file_name=f"ddlt_distribution_{lead_time_days}days.csv",
                        mime="text/csv"
                    )
                        
            # Cost parameters and optimization
            
            st.sidebar.markdown("---")
            st.sidebar.header("💰 Cost Parameters")
            
            with st.sidebar.form(key='enhanced_cost_form'):
                st.subheader("💸 Cost Inputs")
                
                col1, col2 = st.columns(2)
                with col1:
                    cp_cost = st.number_input(
                        "Ordering Cost (฿/order)", 
                        0.0, value=10.0, step=1.0,
                        help="Fixed cost per order placed"
                    )
                    product_cost = st.number_input(
                        "Product Cost (฿/unit)", 
                        0.0, value=50.0, step=1.0,
                        help="Unit cost of the product"
                    )
                
                with col2:
                    h_percent = st.slider(
                        "Holding Rate (%/year)", 
                        0, 50, 10,
                        help="Annual holding cost as % of product cost"
                    )
                    s_percent = st.slider(
                        "Shortage Cost Rate (%)", 
                        0, 100, 30,
                        help="Shortage cost as % of product cost"
                    )

                st.subheader("📋 Inventory Policy")
                case_type = st.radio(
                    "Shortage Handling", 
                    ('Lost Sales', 'Backlog'),
                    help="How shortages are handled in your system"
                )
                
                submitted = st.form_submit_button("🚀 Optimize (Q, R) Policy", type="primary")

        # Main content area for results
        if submitted and 'final_ddlt_with_shortage' in st.session_state.processed_data:
            st.header("🎯 Step 7: Optimal (Q, R) Policy Results")
            create_progress_indicator(7)

            final_ddlt = st.session_state.processed_data['final_ddlt_with_shortage']
            daily_avg_demand = st.session_state.processed_data['final_demand_df']['Total_Demand'].mean()
            
            service_level_type = case_type.lower().replace(' ', '_')

            qr_results = find_optimal_qr(
                ddlt_with_shortage_table=final_ddlt,
                daily_avg_demand=daily_avg_demand,
                avg_lead_time=lead_time_days,
                cp_ordering_cost=cp_cost,
                product_cost=product_cost,
                h_percent_annual=(h_percent / 100.0),
                s_percent=(s_percent / 100.0),
                service_level_type=service_level_type,
                convergence_tolerance=0.01,
                max_iterations=50
            )

            if qr_results:
                st.balloons()
                
                # Results display with enhanced styling
                st.markdown(f"""
                <div class="success-box">
                    <h3>🎉 Optimization Complete - {case_type} Case</h3>
                    <p>Your optimal inventory policy has been calculated!</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Key metrics in styled cards
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    q_rounded = math.ceil(qr_results['optimal_Q'])
                    st.metric(
                        "📦 Order Quantity (Q)", 
                        f"{q_rounded:,}",
                        help="Optimal quantity to order each time (rounded up)"
                    )
                with col2:
                    st.metric(
                        "🔄 Reorder Point (R)", 
                        f"{qr_results['optimal_R']:,.0f}",
                        help="Reorder point to trigger new orders"
                    )
                with col3:
                    st.markdown(f"""<div style='font-size:1.5rem; font-weight:bold; color:#E67E22;'>💰 Total Annual Cost (TAC)</div><div style='font-size:2rem; font-weight:bold;'>{qr_results['min_TAC']:,.2f} ฿</div>""", unsafe_allow_html=True)
                with col4:
                    st.metric(
                        "🔁 Iterations", 
                        f"{qr_results['convergence_iterations']}",
                        help="Number of iterations to converge"
                    )

                # Convergence details
                with st.expander("🔎 View Optimization Details", expanded=False):
                    history_df = qr_results['history']
                    st.dataframe(history_df.style.format({
                        'Q_old': '{:,.2f}',
                        'CSL_optimal': '{:.4f}',
                        'R_found': '{:.0f}',
                        'E_S_at_R': '{:.4f}',
                        'TAC': '{:,.2f}'
                    }))

                    # Visual Analysis of Cost Trade-Offs
                    st.subheader("📈 Cost Component Trade-Offs During Optimization")
                    fig_costs = go.Figure()
                    fig_costs.add_trace(go.Scatter(
                        x=history_df['Iteration'], y=history_df['Ordering_Cost'],
                        mode='lines+markers', name='Ordering Cost',
                        line=dict(color='red', width=2), marker=dict(size=7)
                    ))
                    fig_costs.add_trace(go.Scatter(
                        x=history_df['Iteration'], y=history_df['Holding_Cost'],
                        mode='lines+markers', name='Holding Cost',
                        line=dict(color='gray', width=2), marker=dict(size=7)
                    ))
                    fig_costs.add_trace(go.Scatter(
                        x=history_df['Iteration'], y=history_df['Shortage_Cost'],
                        mode='lines+markers', name='Shortage/Lost Sale Cost',
                        line=dict(color='gold', width=2), marker=dict(size=7)
                    ))
                    fig_costs.add_trace(go.Scatter(
                        x=history_df['Iteration'], y=history_df['TAC'],
                        mode='lines+markers', name='Total Annual Cost (TAC)',
                        line=dict(color='#2980b9', width=3, dash='dash'), marker=dict(size=7)
                    ))
                    fig_costs.update_layout(
                        title='Figure 4.1: Visualizing Cost Component Trade-Offs during Iterative Search',
                        xaxis_title='Iteration',
                        yaxis_title='Cost (฿)',
                        plot_bgcolor='#222c36',
                        paper_bgcolor='#222c36',
                        font_color='white',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                    )
                    st.plotly_chart(fig_costs, use_container_width=True)

# Run the app
if __name__ == "__main__":
    main()
