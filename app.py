import streamlit as st

st.set_page_config(
    page_title = 'Reduction Black Dross / Zone3 Temp Prediction Dashboard',
    layout = 'wide')

import plotly.express as px
import plotly.figure_factory as ff

import pandas as pd
import numpy as np
import pickle

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

## Start of the page
st.title('Real-Time / Z3 Temperature Prediction Dashboard')

st.text(" ")
st.text(" ")
st.text(" ")
st.text(" ")
st.text(" ")
st.text(" ")

# Load z3/z4 Temp Data for distplot
distplot_df = pd.read_pickle('distplot_df.pkl')

# Load PI Data : Recent 4 hrs
PI_df = pd.read_pickle('PI_df.pkl')

# creating a single-element container
placeholder = st.empty()

with placeholder.container():
    # Current Zone3 Temp and Prediction
    Delac_1_z3_now, Delac_1_z3_hat = st.columns(2)
    st.text(" ")
    st.text(" ")
    Delac_2_z3_now, Delac_2_z3_hat = st.columns(2)
    st.text(" ")
    st.text(" ")
    Delac_1_z4_now, Delac_1_z4_hat = st.columns(2)
    st.text(" ")
    st.text(" ")
    Delac_2_z4_now, Delac_2_z4_hat = st.columns(2)
    st.text(" ")
    st.text(" ")

    Delac_1_z3_now.metric(label='Current Delac1 Zone3 Temp 🔥', value=PI_df['Delac_1 WTCT3'].iloc[-1])
    Delac_1_z3_hat.metric(label='Pred Delac1 Zone3 Temp 📈', value=PI_df['Delac_1 Kiln_Temperature_Zone_3_Prediction'].iloc[-1])
    Delac_2_z3_now.metric(label='Current Delac2 Zone3 Temp 🔥', value=PI_df['Delac_2 WTCT3'].iloc[-1])
    Delac_2_z3_hat.metric(label='Pred Delac2 Zone3 Temp 📈', value=PI_df['Delac_2 Kiln_Temperature_Zone_3_Prediction'].iloc[-1])
    Delac_1_z4_now.metric(label='Current Delac1 Zone4 Temp 🔥', value=PI_df['Delac_1 WTCT4'].iloc[-1])
    Delac_1_z4_hat.metric(label='Pred Delac1 Zone4 Temp 📈', value=PI_df['Delac_1 Kiln_Temperature_Zone_4_Prediction'].iloc[-1])
    Delac_2_z4_now.metric(label='Current Delac2 Zone4 Temp 🔥', value=PI_df['Delac_2 WTCT4'].iloc[-1])
    Delac_2_z4_hat.metric(label='Pred Delac2 Zone4 Temp 📈', value=PI_df['Delac_2 Kiln_Temperature_Zone_4_Prediction'].iloc[-1])

    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")

    # create two columns for charts
    Delac_1_z3_true_vs_predict_chart, Delac_2_z3_true_vs_predict_chart = st.columns(2)
    Delac_1_z4_true_vs_predict_chart, Delac_2_z4_true_vs_predict_chart = st.columns(2)

    with Delac_1_z3_true_vs_predict_chart:
        st.markdown("### Delac_1_Z3 True vs Predict")
        fig = px.line(PI_df, x=PI_df.index, y=['Delac_1 WTCT3', 'Delac_1 Kiln_Temperature_Zone_3_Prediction'], color_discrete_map={'Delac_1 WTCT3':'blue', 'Delac_1 Kiln_Temperature_Zone_3_Prediction':'red'})
        st.write(fig)

    with Delac_2_z3_true_vs_predict_chart:
        st.markdown("### Delac_2_Z3 True vs Predict")
        fig = px.line(PI_df, x=PI_df.index, y=['Delac_2 WTCT3', 'Delac_1 Kiln_Temperature_Zone_3_Prediction'], color_discrete_map={'Delac_2 WTCT3':'blue', 'Delac_2 Kiln_Temperature_Zone_3_Prediction':'red'})
        st.write(fig)

    with Delac_1_z4_true_vs_predict_chart:
        st.markdown("### Delac_1_Z4 True vs Predict")
        fig = px.line(PI_df, x=PI_df.index, y=['Delac_1 WTCT4', 'Delac_1 Kiln_Temperature_Zone_4_Prediction'], color_discrete_map={'Delac_1 WTCT4':'blue', 'Delac_1 Kiln_Temperature_Zone_4_Prediction':'red'})
        st.write(fig)

    with Delac_2_z4_true_vs_predict_chart:
        st.markdown("### Delac_2_Z4 True vs Predict")
        fig = px.line(PI_df, x=PI_df.index, y=['Delac_2 WTCT4', 'Delac_2 Kiln_Temperature_Zone_4_Prediction'], color_discrete_map={'Delac_2 WTCT4':'blue', 'Delac_2 Kiln_Temperature_Zone_4_Prediction':'red'})
        st.write(fig)

    Delac_1_z3_standard_deviation_chart, Delac_2_z3_standard_deviation_chart = st.columns(2)
    Delac_1_z4_standard_deviation_chart, Delac_2_z4_standard_deviation_chart = st.columns(2)

    with Delac_1_z3_standard_deviation_chart:
        st.markdown("### Delac_1 z3 Standard deviation")
        hist_data = [distplot_df['Delac_1 WTCT3'], distplot_df['Delac_2 WTCT3']]
        group_labels = ['Before', 'After']
        colors = ['#2BCDC1', '#F66095']
        # Create distplot with curve_type set to 'kde'
        fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors, curve_type='kde')
        st.write(fig)

    with Delac_2_z3_standard_deviation_chart:
        st.markdown("### Delac_2 z3 Standard deviation")
        hist_data = [distplot_df['Delac_2 WTCT3'], distplot_df['Delac_1 WTCT3']]
        group_labels = ['Before', 'After']
        colors = ['#2BCDC1', '#F66095']
        # Create distplot with curve_type set to 'kde'
        fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors, curve_type='kde')
        st.write(fig)

    with Delac_1_z4_standard_deviation_chart:
        st.markdown("### Delac_1 z4 Standard deviation")
        hist_data = [distplot_df['Delac_1 WTCT4'], distplot_df['Delac_2 WTCT4']]
        group_labels = ['Before', 'After']
        colors = ['#2BCDC1', '#F66095']
        # Create distplot with curve_type set to 'kde'
        fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors, curve_type='kde')
        st.write(fig)

    with Delac_2_z4_standard_deviation_chart:
        st.markdown("### Delac_2 z4 Standard deviation")
        hist_data = [distplot_df['Delac_2 WTCT4'], distplot_df['Delac_1 WTCT4']]
        group_labels = ['Before', 'After']
        colors = ['#2BCDC1', '#F66095']
        # Create distplot with curve_type set to 'kde'
        fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors, curve_type='kde')
        st.write(fig)
