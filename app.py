import streamlit as st

st.set_page_config(
    page_title = 'Reduction Black Dross / Zone3 Temp Prediction Dashboard',
    layout = 'wide')

import os
# from multiprocessing import Process

import plotly.express as px
import plotly.figure_factory as ff

import time
import schedule
import datetime
from datetime import date
import PIconnect as PI

import pandas as pd
import numpy as np

import itertools

import pickle

import seaborn as sns
from matplotlib import pyplot as plt

import missingno as msno

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


def print5min():
    print("5 min!")

schedule.every(5).minutes.do(print5min)

####################################################################################
## Check Server Status
@st.cache(allow_output_mutation=True)
def find_server():
    with PI.PIServer() as server:
        return server.server_name
print('server name:', find_server())

## Data Transform
@st.cache(allow_output_mutation=True)
def data_to_python(tag_list, start, end, freq):
    with PI.PIServer() as server:
        data_all = pd.DataFrame()

        for tag in tag_list:
            points = server.search(tag)[0]
            data = points.interpolated_values(start_time=start, end_time=end, interval=freq)
            data = pd.to_numeric(data, errors='coerce')
            data_all = pd.concat([data_all, data], axis=1)
    return data_all

tag_list = ['Delac_1 WTCT3'
,'Delac_2 WTCT3'
,'Delac_1 WTCT4'
,'Delac_2 WTCT4'
,'Delac_1 Kiln_Temperature_Zone_3_Prediction'
,'Delac_2 Kiln_Temperature_Zone_3_Prediction'
,'Delac_1 Kiln_Temperature_Zone_4_Prediction'
,'Delac_2 Kiln_Temperature_Zone_4_Prediction']

PI.PIConfig.DEFAULT_TIMEZONE = 'Asia/Seoul'
####################################################################################

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

# creating a single-element container
placeholder = st.empty()

while True:

    today = date.today()
    tmr = today + datetime.timedelta(days=1)
    today = today.strftime("%Y-%m-%d")
    tmr = tmr.strftime("%Y-%m-%d")

    PI_df = data_to_python(tag_list, today, tmr, '1m')
    PI_df.index = pd.to_datetime(PI_df.index).tz_localize(None)
    PI_df = PI_df.dropna()
    PI_df = PI_df.tail(240)


    with placeholder.container():

        st.dataframe(PI_df)

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

    schedule.run_pending()
