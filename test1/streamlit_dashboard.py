import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide")

st.title("🤖 AI-Driven SAN Traffic Management Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    try:
        before_df = pd.read_csv('san_traffic.csv')
        after_df = pd.read_csv('optimized_traffic.csv')
        return before_df, after_df
    except FileNotFoundError:
        return None, None

before_df, after_df = load_data()

if before_df is None or after_df is None:
    st.error("Data files not found. Please run all preceding scripts first: `san_traffic_simulator.py`, `ml_model_trainer.py`, and `san_traffic_optimizer.py`.")
else:
    st.success("Successfully loaded simulation and optimization data.")

    # --- Phase 5: Visualization Dashboard ---

    st.header("Traffic Optimization: Before vs. After")

    # ----- KPI Metrics -----
    st.subheader("Key Performance Indicators (KPIs)")
    
    # Calculate KPIs
    avg_latency_before = before_df['Latency_ms'].mean()
    avg_latency_after = after_df['Latency_ms'].mean()
    latency_reduction = ((avg_latency_before - avg_latency_after) / avg_latency_before) * 100

    congestion_time_before = before_df['Congestion'].sum()
    congestion_time_after = after_df['Congestion'].sum()
    
    # FIX: Add a check to prevent division by zero if congestion_time_before is 0
    if congestion_time_before > 0:
        congestion_reduction = ((congestion_time_before - congestion_time_after) / congestion_time_before) * 100
    else:
        congestion_reduction = 0

    actions_taken = after_df[after_df['Action_Taken'] != 'None'].shape[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg. Latency (Before)", f"{avg_latency_before:.2f} ms")
    col1.metric("Avg. Latency (After)", f"{avg_latency_after:.2f} ms", f"{latency_reduction:.2f}% Reduction", delta_color="inverse")
    
    col2.metric("Time in Congestion (Before)", f"{congestion_time_before} steps")
    col2.metric("Time in Congestion (After)", f"{congestion_time_after} steps", f"{congestion_reduction:.2f}% Reduction", delta_color="inverse")

    col3.metric("Total IOPS (Before)", f"{before_df['Read_IOPS'].sum() + before_df['Write_IOPS'].sum():,}")
    col3.metric("Total IOPS (After)", f"{after_df['Read_IOPS'].sum() + after_df['Write_IOPS'].sum():,}")
    col3.metric("Optimization Actions Taken", f"{actions_taken} times")


    # ----- Time Series Charts -----
    st.subheader("Latency & IOPS Over Time")

    # Latency Chart
    fig_latency = go.Figure()
    fig_latency.add_trace(go.Scatter(x=before_df['Time'], y=before_df['Latency_ms'], name='Latency (Before)', line=dict(color='rgba(236, 8, 8, 0.5)')))
    fig_latency.add_trace(go.Scatter(x=after_df['Time'], y=after_df['Latency_ms'], name='Latency (After Optimization)', line=dict(color='rgba(8, 236, 8, 1)')))
    
    fig_latency.add_hline(y=10, line_dash="dot", line_color="gray", annotation_text="Congestion Threshold (10ms)")
    
    fig_latency.update_layout(title='Latency Before vs. After Optimization',
                              xaxis_title='Time Step',
                              yaxis_title='Latency (ms)',
                              legend_title='Legend')
    # FIX: Replaced use_container_width=True with width='stretch'
    st.plotly_chart(fig_latency, width='stretch')


    # IOPS Chart
    fig_iops = go.Figure()
    
    # --- THESE ARE THE CHANGED LINES ---
    # Old "Before" color: 'rgba(66, 135, 245, 0.5)' (light blue)
    # New "Before" color: 'rgba(255, 165, 0, 0.7)' (Orange)
    fig_iops.add_trace(go.Scatter(x=before_df['Time'], y=before_df['Read_IOPS'] + before_df['Write_IOPS'], name='Total IOPS (Before)', line=dict(color='rgba(255, 165, 0, 0.7)')))
    
    # Old "After" color: 'rgba(11, 64, 130, 1)' (dark blue)
    # New "After" color: 'rgba(0, 191, 255, 1)' (Deep Sky Blue)
    fig_iops.add_trace(go.Scatter(x=after_df['Time'], y=after_df['Read_IOPS'] + after_df['Write_IOPS'], name='Total IOPS (After Optimization)', line=dict(color='rgba(0, 191, 255, 1)')))
    
    fig_iops.update_layout(title='Total IOPS Before vs. After Optimization',
                           xaxis_title='Time Step',
                           yaxis_title='Total IOPS',
                           legend_title='Legend')
    # FIX: Replaced use_container_width=True with width='stretch'
    st.plotly_chart(fig_iops, width='stretch')

    # ----- Pie Chart -----
    st.subheader("Congestion Analysis")
    
    pie_data = pd.DataFrame({
        'State': ['Congested', 'Not Congested', 'Congested', 'Not Congested'],
        'Time Steps': [congestion_time_before, len(before_df) - congestion_time_before,
                       congestion_time_after, len(after_df) - congestion_time_after],
        'Group': ['Before Optimization', 'Before Optimization', 'After Optimization', 'After Optimization']
    })

    fig_pie = px.pie(pie_data, values='Time Steps', names='State', 
                     color='State',
                     color_discrete_map={'Congested':'red', 'Not Congested':'green'},
                     facet_col='Group',
                     title='% Time Spent in Congestion')
    
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    # FIX: Replaced use_container_width=True with width='stretch'
    st.plotly_chart(fig_pie, width='stretch')

    # ----- Data Inspector -----
    st.subheader("Data Inspector")
    with st.expander("Show Raw Simulation Data"):
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(before_df.head(100))
        with col2:
            st.dataframe(after_df.head(100))

