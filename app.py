import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page config for wide layout
st.set_page_config(layout="wide", page_title="Financial Model Dashboard")

# Custom CSS for better styling
st.markdown("""
<style>
    .reportview-container .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e0e0e0;
    }
    .small-font {
        font-size: 0.8rem;
    }
    .st-emotion-cache-10trblm {
        position: relative;
        left: 0px;
        top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Load the data
@st.cache_data
def load_data():
    data = {
        'Month': list(range(1, 25)),
        'Total MRR': [8823, 8823, 11077, 13331, 15585, 17840, 20094, 22348, 24602, 26857, 29111, 31365, 35391, 39416, 43442, 47467, 51493, 55518, 59544, 63569, 67595, 71620, 75646, 79671],
        'Monthly Burn': [130000] * 24,
        'One Time Expenses': [0, 0, 56300, 50000] + [0] * 20,
        'Actual Burn': [121177, 121177, 118922, 116668, 114414, 112159, 109905, 107651, 105397, 103142, 100888, 98634, 94608, 90583, 86557, 82532, 78506, 74481, 70455, 66430, 62404, 58379, 54353, 50328],
        'Cash Balance': [1878823, 1757646, 1582423, 1415754, 1301340, 1189180, 1079275, 971623, 866226, 763084, 662195, 563561, 568813, 478230, 391672, 309140, 230633, 156152, 85696, 19266, -43138, -101517, -155871, -206199]
    }
    df = pd.DataFrame(data)
    df['MRR Growth Rate'] = df['Total MRR'].pct_change()
    df['Burn Rate'] = df['Actual Burn'] / df['Cash Balance']
    df['Runway (Months)'] = df['Cash Balance'] / df['Actual Burn']
    return df

df = load_data()

# Initialize session state for scenarios
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = {
        'Expected': {'growth_rate': float(df['MRR Growth Rate'].mean()), 'burn_reduction': 1.0},
        'Optimistic': {'growth_rate': min(float(df['MRR Growth Rate'].mean()) * 1.5, 0.5), 'burn_reduction': 0.9},
        'Pessimistic': {'growth_rate': max(float(df['MRR Growth Rate'].mean()) * 0.5, 0.01), 'burn_reduction': 1.1}
    }

# Function to calculate projections
def calculate_projections(df, projection_months, growth_rate, burn_reduction):
    last_month = df.iloc[-1]
    projections = pd.DataFrame({
        'Month': range(25, 25 + projection_months),
        'Total MRR': [last_month['Total MRR'] * (1 + growth_rate) ** i for i in range(1, projection_months + 1)],
        'Monthly Burn': [last_month['Monthly Burn'] * burn_reduction] * projection_months,
        'One Time Expenses': [0] * projection_months
    })
    projections['Actual Burn'] = projections['Monthly Burn'] - projections['Total MRR']
    projections['Cash Balance'] = [last_month['Cash Balance']] + [0] * (projection_months - 1)
    for i in range(1, projection_months):
        projections.loc[projections.index[i], 'Cash Balance'] = (
            projections.loc[projections.index[i-1], 'Cash Balance'] - 
            projections.loc[projections.index[i-1], 'Actual Burn']
        )
    projections['MRR Growth Rate'] = growth_rate
    projections['Burn Rate'] = projections['Actual Burn'] / projections['Cash Balance']
    projections['Runway (Months)'] = projections['Cash Balance'] / projections['Actual Burn']
    return pd.concat([df, projections], ignore_index=True)

# Main app
st.title('Financial Model Dashboard')

# Tabs
tab1, tab2 = st.tabs(["Dashboard", "Edit Scenarios"])

with tab1:
    # Sidebar for user input
    st.sidebar.header('Projections and Scenarios')
    projection_months = st.sidebar.slider('Number of months to project', 1, 36, 12)
    scenario = st.sidebar.selectbox('Select Scenario', ['Expected', 'Optimistic', 'Pessimistic'])

    # Calculate projections based on selected scenario
    growth_rate = st.session_state.scenarios[scenario]['growth_rate']
    burn_reduction = st.session_state.scenarios[scenario]['burn_reduction']
    df_combined = calculate_projections(df, projection_months, growth_rate, burn_reduction)

    # Metrics
    st.header('Key Metrics')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric('Current MRR', f'${df["Total MRR"].iloc[-1]:,.0f}')
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric('Current Burn Rate', f'{df["Burn Rate"].iloc[-1]:.2%}')
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric('Current Runway', f'{df["Runway (Months)"].iloc[-1]:.1f} months')
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric('Projected MRR (End of Period)', f'${df_combined["Total MRR"].iloc[-1]:,.0f}')
        st.markdown('</div>', unsafe_allow_html=True)

    # Interactive Charts
    st.header('Financial Charts')

    col1, col2 = st.columns(2)

    with col1:
        # MRR and Burn Rate
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df_combined['Month'], y=df_combined['Total MRR'], name="MRR", line=dict(color="#1f77b4", width=3)))
        fig.add_trace(go.Scatter(x=df_combined['Month'], y=df_combined['Actual Burn'], name="Burn", line=dict(color="#d62728", width=3)), secondary_y=True)
        fig.update_layout(title='MRR vs Burn Rate Over Time', template="plotly_white", height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_xaxes(title_text='Month')
        fig.update_yaxes(title_text='Total MRR ($)', secondary_y=False)
        fig.update_yaxes(title_text='Actual Burn ($)', secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        # MRR Growth Rate
        fig = px.line(df_combined, x='Month', y='MRR Growth Rate', title='MRR Growth Rate Over Time')
        fig.update_layout(template="plotly_white", height=400)
        fig.update_traces(line=dict(width=3, color="#2ca02c"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Cash Balance
        fig = px.line(df_combined, x='Month', y='Cash Balance', title='Cash Balance Over Time')
        fig.update_layout(template="plotly_white", height=400)
        fig.update_traces(line=dict(width=3, color="#ff7f0e"))
        st.plotly_chart(fig, use_container_width=True)

        # Runway Projection
        fig = px.line(df_combined, x='Month', y='Runway (Months)', title='Runway Projection')
        fig.update_layout(template="plotly_white", height=400)
        fig.update_traces(line=dict(width=3, color="#9467bd"))
        st.plotly_chart(fig, use_container_width=True)

    # Break-even Analysis
    break_even_month = df_combined[df_combined['Total MRR'] >= df_combined['Monthly Burn']]['Month'].min()
    if pd.notna(break_even_month):
        st.success(f"Projected break-even point: Month {break_even_month}")
    else:
        st.warning("Break-even point not reached in the projected period.")

    # Sensitivity Analysis
    st.header('Sensitivity Analysis')
    sensitivity_growth_rates = [growth_rate * 0.5, growth_rate, growth_rate * 1.5]
    sensitivity_data = []

    for sens_growth_rate in sensitivity_growth_rates:
        last_mrr = df['Total MRR'].iloc[-1]
        projected_mrr = last_mrr * (1 + sens_growth_rate) ** projection_months
        sensitivity_data.append({'Growth Rate': f'{sens_growth_rate:.1%}', 'Projected MRR': projected_mrr})

    sensitivity_df = pd.DataFrame(sensitivity_data)
    fig = px.bar(sensitivity_df, x='Growth Rate', y='Projected MRR', title='MRR Sensitivity to Growth Rate')
    fig.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Download link for the data
    csv = df_combined.to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="financial_model_data.csv",
        mime="text/csv",
    )

    # Assumptions and Notes
    st.header('Assumptions and Notes')
    st.markdown("""
    - The model assumes a constant burn rate, adjusted by the selected scenario.
    - One-time expenses are not projected into the future.
    - The break-even point is calculated based on when MRR exceeds the monthly burn.
    - Sensitivity analysis shows the impact of different growth rates on projected MRR.
    """)

    st.markdown('<p class="small-font">Note: This is a simplified model and should be used for illustrative purposes only. Consult with financial advisors for investment decisions.</p>', unsafe_allow_html=True)

with tab2:
    st.header("Edit Scenario Parameters")
    
    for scenario in ['Expected', 'Optimistic', 'Pessimistic']:
        st.subheader(f"{scenario} Scenario")
        col1, col2 = st.columns(2)
        with col1:
            growth_rate = st.number_input(f"Growth Rate for {scenario}", 
                                          min_value=0.0, 
                                          max_value=1.0, 
                                          value=st.session_state.scenarios[scenario]['growth_rate'], 
                                          step=0.01, 
                                          format="%.2f",
                                          key=f"{scenario}_growth")
        with col2:
            burn_reduction = st.number_input(f"Burn Reduction Factor for {scenario}", 
                                             min_value=0.1, 
                                             max_value=2.0, 
                                             value=st.session_state.scenarios[scenario]['burn_reduction'], 
                                             step=0.1, 
                                             format="%.1f",
                                             key=f"{scenario}_burn")
        
        st.session_state.scenarios[scenario]['growth_rate'] = growth_rate
        st.session_state.scenarios[scenario]['burn_reduction'] = burn_reduction
        
    st.markdown("Note: Changes made here will be reflected in the Dashboard tab.")