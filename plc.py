import os
# from multiprocessing import Process

from PIL import Image

import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff

import time
import datetime
from datetime import date
import PIconnect as PI
from pylogix import PLC

import pandas as pd
import numpy as np

import itertools

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

import pickle
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import math

import seaborn as sns
from matplotlib import pyplot as plt

import missingno as msno

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


####################################################################################
## Start of the page
# novelis_logo = Image.open("novelis.png")
#page_icon = novelis_logo,
st.set_page_config(
    page_title = 'Reduction Black Dross / Zone3 Temp Prediction Dashboard',
    layout = 'wide'
)

st.title('Real-Time / Z3 Temperature Prediction Dashboard')

st.text(" ")
st.text(" ")
st.text(" ")
st.text(" ")
st.text(" ")
st.text(" ")

# Load z3 Data for distplot
z3_distplot = pd.read_pickle('z3_distplot.pkl')

# creating a single-element container
placeholder = st.empty()

with PLC() as Delac_1_comm, PLC() as Delac_2_comm:

    Delac_1_comm.IPAddress = '10.44.192.50' # Decoater 1
    Delac_1_comm.SetPLCTime()
    Delac_2_comm.IPAddress = '10.44.192.70' # Decoater 2
    Delac_2_comm.SetPLCTime()

    Delac_1_plc_df = pd.DataFrame()
    Delac_2_plc_df = pd.DataFrame()

    while True:
        ###############################################
        time_now = pd.Series(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), name='plc_time')
        Delac_1_z3_temperature = pd.Series(Delac_1_comm.Read('Kiln_Temperature_Zone_3').Value, name='Delac_1 z3', dtype='float')
        Delac_2_z3_temperature = pd.Series(Delac_2_comm.Read('Kiln_Temperature_Zone_3').Value, name='Delac_2 z3', dtype='float')
        Delac_1_z3_pred = pd.Series(Delac_1_comm.Read('Kiln_Temperature_Zone_3_Prediction').Value, name='Delac_1 z3 Pred', dtype='float')
        Delac_2_z3_pred = pd.Series(Delac_2_comm.Read('Kiln_Temperature_Zone_3_Prediction').Value, name='Delac_2 z3 Pred', dtype='float')

        Delac_1_plc_temp = pd.concat([time_now, Delac_1_z3_temperature, Delac_1_z3_pred], axis=1)
        Delac_2_plc_temp = pd.concat([time_now, Delac_2_z3_temperature, Delac_2_z3_pred], axis=1)

        Delac_1_plc_df = pd.concat([Delac_1_plc_df, Delac_1_plc_temp], axis=0)
        Delac_2_plc_df = pd.concat([Delac_2_plc_df, Delac_2_plc_temp], axis=0)

        Delac_1_plc_df = Delac_1_plc_df.tail(1500)
        Delac_2_plc_df = Delac_2_plc_df.tail(1500)
        ###############################################

        with placeholder.container():

            # Current Zone3 Temp and Prediction
            Delac_1_z3_now, Delac_2_z3_now = st.columns(2)
            st.text(" ")
            st.text(" ")
            Delac_1_z3_hat, Delac_2_z3_hat = st.columns(2)
            st.text(" ")
            st.text(" ")

            Delac_1_z3_now.metric(label='Current Delac1 Zone3 Temp 🔥', value=round(Delac_1_plc_df['Delac_1 z3'].iloc[-1], 1)) # delta=round(Delac_1_plc_df['Delac_1 z3'].iloc[-1] - Delac_1_plc_df['Delac_1 z3'].iloc[-2]
            Delac_1_z3_hat.metric(label='Pred Delac1 Zone3 Temp 📈', value=round(Delac_1_plc_df['Delac_1 z3 Pred'].iloc[-1], 1)) # delta=round(Delac_1_plc_df['Delac_1 z3 Pred'].iloc[-1] - Delac_1_plc_df['Delac_1 z3 Pred'].iloc[-2]
            Delac_2_z3_now.metric(label='Current Delac2 Zone3 Temp 🔥', value=round(Delac_2_plc_df['Delac_2 z3'].iloc[-1], 1)) # delta=round(Delac_2_plc_df['Delac_2 z3'].iloc[-1] - Delac_2_plc_df['Delac_2 z3'].iloc[-2]
            Delac_2_z3_hat.metric(label='Pred Delac2 Zone3 Temp 📈', value=round(Delac_2_plc_df['Delac_2 z3 Pred'].iloc[-1], 1)) # delta=round(Delac_2_plc_df['Delac_2 z3 Pred'].iloc[-1] - Delac_2_plc_df['Delac_2 z3 Pred'].iloc[-2]

            # create two columns for charts
            st.text(" ")
            st.text(" ")
            st.text(" ")
            st.text(" ")
            st.text(" ")
            st.text(" ")

            Delac_1_true_vs_predict_chart, Delac_2_true_vs_predict_chart = st.columns(2)

            with Delac_1_true_vs_predict_chart:
                st.markdown("### Comparison on Delac_1 True vs Predict")
                fig = px.line(Delac_1_plc_df, x='plc_time', y=['Delac_1 z3', 'Delac_1 z3 Pred'], color_discrete_map={'Delac_1 z3':'blue', 'Delac_1 z3 Pred':'red'})
                st.write(fig)

            with Delac_2_true_vs_predict_chart:
                st.markdown("### Comparison on Delac_2 True vs Predict")
                fig = px.line(Delac_2_plc_df, x='plc_time', y=['Delac_2 z3', 'Delac_2 z3 Pred'], color_discrete_map={'Delac_2 z3':'blue', 'Delac_2 z3 Pred':'red'})
                st.write(fig)

            Delac_1_standard_deviation_chart, Delac_2_standard_deviation_chart = st.columns(2)

            with Delac_1_standard_deviation_chart:
                st.markdown("### Comparison on   Delac_1   Standard deviation")
                hist_data = [z3_distplot['Delac_1 WTCT3'], z3_distplot['Delac_2 WTCT3']]
                group_labels = ['Delac_1 z3 Std', 'Delac_2 z3 Std']
                colors = ['#2BCDC1', '#F66095']
                # Create distplot with curve_type set to 'kde'
                fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors, curve_type='kde')
                st.write(fig)

            with Delac_2_standard_deviation_chart:
                st.markdown("### Comparison on   Delac_2   Standard deviation")
                hist_data = [z3_distplot['Delac_1 WTCT3'], z3_distplot['Delac_2 WTCT3']]
                group_labels = ['Delac_1 z3 Std', 'Delac_2 z3 Std']
                colors = ['#2BCDC1', '#F66095']
                # Create distplot with curve_type set to 'kde'
                fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors, curve_type='kde')
                st.write(fig)

        time.sleep(60 - time.time() % 15)
##############################################################



##############################################################
