import streamlit as st
import plotly.express

import datetime
import PIconnect as PI

import pandas as pd
import numpy as np

import itertools

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfullerㅁ
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

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

##############################################################

## Check Server Status

def find_server():
    with PI.PIServer() as server:
        return server.server_name


print('server name:', find_server())

## Data Transform

def data_to_python(tag_list, start, end, freq):
    with PI.PIServer() as server:
        data_all = pd.DataFrame()

        for tag in tag_list:
            points = server.search(tag)[0]
            data = points.interpolated_values(start_time=start, end_time=end, interval=freq)
            data = pd.to_numeric(data, errors='coerce')
            data_all = pd.concat([data_all, data], axis=1)
    return data_all

tag_list = ['Delac_1 Conveyor_Feedrate_PV'
,'Delac_1 Kiln_Discharge_O2'
,'Delac_1 WTCT4'
,'Delac_1 WTCT3'
,'Delac_1 Kiln_Gas_Inlet_Temp'
,'Delac_1 Recirc_Fan_Motor_Requested_Speed'
,'Delac_1 Diverter_Valve_Control_Valve_Feedback'
,'Delac_1 Afterburner_O2'
,'Delac_1 Afterburner_Temp'
,'Delac_1 Afterburner_Temperature_Setpoint'
,'Delac_1 Excess_Air_Valve_Motor_Control_Signal'
,'Delac_1 Recirculation_Fan_Inlet_Temp'
,'Delac_1 Kiln_Excess_Air_Valve_Feedback'
,'Delac_1 Constant_Ideal_Kiln_Temp_Setpoint'
,'Delac_1 WaterFlow_Kiln'
,'Delac_1 WaterFlow_Afterburner'
,'Delac_1 WaterFlow_Duct']

PI.PIConfig.DEFAULT_TIMEZONE = 'Asia/Seoul'

df = data_to_python(tag_list, '2022-04-01', '2022-04-30', '1m')
df.index = pd.to_datetime(df.index).tz_localize(None)

df['Z4_Setpoint_Delta'] = df['Delac_1 Constant_Ideal_Kiln_Temp_Setpoint'] - df['Delac_1 WTCT4']

# MON : 0, Tue : 1, Wed : 2, Thurs : 3, Fri : 4, Sat : 5, Sun : 6
df['dayofweek'] = df.index.dayofweek
df = df[(df['dayofweek'] != 1)]

df['Delac_1 WTCT3'].describe(percentiles=[0.05, 0.95])
df['normal'] = np.where(df['Delac_1 WTCT3'].between(389.622300, 427.017200), 1, 0)
df.resample('D')['normal'].mean().sort_values(ascending=False).head(10)

s1 = '2022-04-03'
s2 = '2022-04-04'
s3 = '2022-04-06'
s4 = '2022-04-10'
s5 = '2022-04-13'
s6 = '2022-04-14'
s7 = '2022-04-18'
s8 = '2022-04-21'
s9 = '2022-04-28'
s10 = '2022-04-30'

mdl_data = pd.concat([df.loc[s1], df.loc[s2], df.loc[s3], df.loc[s4], df.loc[s5], df.loc[s6], df.loc[s7], df.loc[s8], df.loc[s9], df.loc[s10]], axis=0)

var_list = ['Delac_1 WTCT4', 'Delac_1 WTCT3', 'Delac_1 Recirc_Fan_Motor_Requested_Speed', 'Delac_1 Diverter_Valve_Control_Valve_Feedback', 'Delac_1 Conveyor_Feedrate_PV', 'Delac_1 Kiln_Discharge_O2', 'Delac_1 Kiln_Gas_Inlet_Temp', 'Delac_1 Afterburner_O2', 'Delac_1 Afterburner_Temp', 'Delac_1 Kiln_Excess_Air_Valve_Feedback', 'Z4_Setpoint_Delta']
mdl_data = mdl_data[var_list].copy()

for var in var_list:
    mdl_data[var + '_dif1'] = mdl_data[var] - mdl_data[var].shift(1)
    mdl_data[var + '_dif1_shift1'] = mdl_data[var].shift(1) - mdl_data[var].shift(2)
    mdl_data[var + '_dif1_shift2'] = mdl_data[var].shift(2) - mdl_data[var].shift(3)
    mdl_data[var + '_dif1_shift3'] = mdl_data[var].shift(3) - mdl_data[var].shift(4)
    mdl_data[var + '_dif1_shift4'] = mdl_data[var].shift(4) - mdl_data[var].shift(5)
    mdl_data[var + '_dif1_shift5'] = mdl_data[var].shift(5) - mdl_data[var].shift(6)
    mdl_data[var + '_dif1_shift6'] = mdl_data[var].shift(6) - mdl_data[var].shift(7)
    mdl_data[var + '_dif1_shift7'] = mdl_data[var].shift(7) - mdl_data[var].shift(8)
    mdl_data[var + '_dif1_shift8'] = mdl_data[var].shift(8) - mdl_data[var].shift(9)
    mdl_data[var + '_dif1_shift9'] = mdl_data[var].shift(9) - mdl_data[var].shift(10)

    mdl_data[var + '_dif2'] = mdl_data[var] - mdl_data[var].shift(2)
    mdl_data[var + '_dif2_shift1'] = mdl_data[var].shift(1) - mdl_data[var].shift(3)
    mdl_data[var + '_dif2_shift2'] = mdl_data[var].shift(2) - mdl_data[var].shift(4)
    mdl_data[var + '_dif2_shift3'] = mdl_data[var].shift(3) - mdl_data[var].shift(5)
    mdl_data[var + '_dif2_shift4'] = mdl_data[var].shift(4) - mdl_data[var].shift(6)
    mdl_data[var + '_dif2_shift5'] = mdl_data[var].shift(5) - mdl_data[var].shift(7)
    mdl_data[var + '_dif2_shift6'] = mdl_data[var].shift(6) - mdl_data[var].shift(8)
    mdl_data[var + '_dif2_shift7'] = mdl_data[var].shift(7) - mdl_data[var].shift(9)
    mdl_data[var + '_dif2_shift8'] = mdl_data[var].shift(8) - mdl_data[var].shift(10)
    mdl_data[var + '_dif2_shift9'] = mdl_data[var].shift(9) - mdl_data[var].shift(11)

    mdl_data[var + '_dif3'] = mdl_data[var] - mdl_data[var].shift(3)
    mdl_data[var + '_dif3_shift1'] = mdl_data[var].shift(1) - mdl_data[var].shift(4)
    mdl_data[var + '_dif3_shift2'] = mdl_data[var].shift(2) - mdl_data[var].shift(5)
    mdl_data[var + '_dif3_shift3'] = mdl_data[var].shift(3) - mdl_data[var].shift(6)
    mdl_data[var + '_dif3_shift4'] = mdl_data[var].shift(4) - mdl_data[var].shift(7)
    mdl_data[var + '_dif3_shift5'] = mdl_data[var].shift(5) - mdl_data[var].shift(8)
    mdl_data[var + '_dif3_shift6'] = mdl_data[var].shift(6) - mdl_data[var].shift(9)
    mdl_data[var + '_dif3_shift7'] = mdl_data[var].shift(7) - mdl_data[var].shift(10)
    mdl_data[var + '_dif3_shift8'] = mdl_data[var].shift(8) - mdl_data[var].shift(11)
    mdl_data[var + '_dif3_shift9'] = mdl_data[var].shift(9) - mdl_data[var].shift(12)

    mdl_data[var + '_dif4'] = mdl_data[var] - mdl_data[var].shift(4)
    mdl_data[var + '_dif4_shift1'] = mdl_data[var].shift(1) - mdl_data[var].shift(5)
    mdl_data[var + '_dif4_shift2'] = mdl_data[var].shift(2) - mdl_data[var].shift(6)
    mdl_data[var + '_dif4_shift3'] = mdl_data[var].shift(3) - mdl_data[var].shift(7)
    mdl_data[var + '_dif4_shift4'] = mdl_data[var].shift(4) - mdl_data[var].shift(8)
    mdl_data[var + '_dif4_shift5'] = mdl_data[var].shift(5) - mdl_data[var].shift(9)
    mdl_data[var + '_dif4_shift6'] = mdl_data[var].shift(6) - mdl_data[var].shift(10)
    mdl_data[var + '_dif4_shift7'] = mdl_data[var].shift(7) - mdl_data[var].shift(11)
    mdl_data[var + '_dif4_shift8'] = mdl_data[var].shift(8) - mdl_data[var].shift(12)
    mdl_data[var + '_dif4_shift9'] = mdl_data[var].shift(9) - mdl_data[var].shift(13)

    mdl_data[var + '_dif5'] = mdl_data[var] - mdl_data[var].shift(5)
    mdl_data[var + '_dif5_shift1'] = mdl_data[var].shift(1) - mdl_data[var].shift(6)
    mdl_data[var + '_dif5_shift2'] = mdl_data[var].shift(2) - mdl_data[var].shift(7)
    mdl_data[var + '_dif5_shift3'] = mdl_data[var].shift(3) - mdl_data[var].shift(8)
    mdl_data[var + '_dif5_shift4'] = mdl_data[var].shift(4) - mdl_data[var].shift(9)
    mdl_data[var + '_dif5_shift5'] = mdl_data[var].shift(5) - mdl_data[var].shift(10)
    mdl_data[var + '_dif5_shift6'] = mdl_data[var].shift(6) - mdl_data[var].shift(11)
    mdl_data[var + '_dif5_shift7'] = mdl_data[var].shift(7) - mdl_data[var].shift(12)
    mdl_data[var + '_dif5_shift8'] = mdl_data[var].shift(8) - mdl_data[var].shift(13)
    mdl_data[var + '_dif5_shift9'] = mdl_data[var].shift(9) - mdl_data[var].shift(14)

    mdl_data[var + '_dif6'] = mdl_data[var] - mdl_data[var].shift(6)
    mdl_data[var + '_dif6_shift1'] = mdl_data[var].shift(1) - mdl_data[var].shift(7)
    mdl_data[var + '_dif6_shift2'] = mdl_data[var].shift(2) - mdl_data[var].shift(8)
    mdl_data[var + '_dif6_shift3'] = mdl_data[var].shift(3) - mdl_data[var].shift(9)
    mdl_data[var + '_dif6_shift4'] = mdl_data[var].shift(4) - mdl_data[var].shift(10)
    mdl_data[var + '_dif6_shift5'] = mdl_data[var].shift(5) - mdl_data[var].shift(11)
    mdl_data[var + '_dif6_shift6'] = mdl_data[var].shift(6) - mdl_data[var].shift(12)
    mdl_data[var + '_dif6_shift7'] = mdl_data[var].shift(7) - mdl_data[var].shift(13)
    mdl_data[var + '_dif6_shift8'] = mdl_data[var].shift(8) - mdl_data[var].shift(14)
    mdl_data[var + '_dif6_shift9'] = mdl_data[var].shift(9) - mdl_data[var].shift(15)

    mdl_data[var + '_dif7'] = mdl_data[var] - mdl_data[var].shift(7)
    mdl_data[var + '_dif7_shift1'] = mdl_data[var].shift(1) - mdl_data[var].shift(8)
    mdl_data[var + '_dif7_shift2'] = mdl_data[var].shift(2) - mdl_data[var].shift(9)
    mdl_data[var + '_dif7_shift3'] = mdl_data[var].shift(3) - mdl_data[var].shift(10)
    mdl_data[var + '_dif7_shift4'] = mdl_data[var].shift(4) - mdl_data[var].shift(11)
    mdl_data[var + '_dif7_shift5'] = mdl_data[var].shift(5) - mdl_data[var].shift(12)
    mdl_data[var + '_dif7_shift6'] = mdl_data[var].shift(6) - mdl_data[var].shift(13)
    mdl_data[var + '_dif7_shift7'] = mdl_data[var].shift(7) - mdl_data[var].shift(14)
    mdl_data[var + '_dif7_shift8'] = mdl_data[var].shift(8) - mdl_data[var].shift(15)
    mdl_data[var + '_dif7_shift9'] = mdl_data[var].shift(9) - mdl_data[var].shift(16)

    mdl_data[var + '_dif8'] = mdl_data[var] - mdl_data[var].shift(8)
    mdl_data[var + '_dif8_shift1'] = mdl_data[var].shift(1) - mdl_data[var].shift(9)
    mdl_data[var + '_dif8_shift2'] = mdl_data[var].shift(2) - mdl_data[var].shift(10)
    mdl_data[var + '_dif8_shift3'] = mdl_data[var].shift(3) - mdl_data[var].shift(11)
    mdl_data[var + '_dif8_shift4'] = mdl_data[var].shift(4) - mdl_data[var].shift(12)
    mdl_data[var + '_dif8_shift5'] = mdl_data[var].shift(5) - mdl_data[var].shift(13)
    mdl_data[var + '_dif8_shift6'] = mdl_data[var].shift(6) - mdl_data[var].shift(14)
    mdl_data[var + '_dif8_shift7'] = mdl_data[var].shift(7) - mdl_data[var].shift(15)
    mdl_data[var + '_dif8_shift8'] = mdl_data[var].shift(8) - mdl_data[var].shift(16)
    mdl_data[var + '_dif8_shift9'] = mdl_data[var].shift(9) - mdl_data[var].shift(17)

    mdl_data[var + '_dif9'] = mdl_data[var] - mdl_data[var].shift(9)
    mdl_data[var + '_dif9_shift1'] = mdl_data[var].shift(1) - mdl_data[var].shift(10)
    mdl_data[var + '_dif9_shift2'] = mdl_data[var].shift(2) - mdl_data[var].shift(11)
    mdl_data[var + '_dif9_shift3'] = mdl_data[var].shift(3) - mdl_data[var].shift(12)
    mdl_data[var + '_dif9_shift4'] = mdl_data[var].shift(4) - mdl_data[var].shift(13)
    mdl_data[var + '_dif9_shift5'] = mdl_data[var].shift(5) - mdl_data[var].shift(14)
    mdl_data[var + '_dif9_shift6'] = mdl_data[var].shift(6) - mdl_data[var].shift(15)
    mdl_data[var + '_dif9_shift7'] = mdl_data[var].shift(7) - mdl_data[var].shift(16)
    mdl_data[var + '_dif9_shift8'] = mdl_data[var].shift(8) - mdl_data[var].shift(17)
    mdl_data[var + '_dif9_shift9'] = mdl_data[var].shift(9) - mdl_data[var].shift(18)


mdl_data['target_5'] = mdl_data['Delac_1 WTCT3'].shift(-5) - mdl_data['Delac_1 WTCT3']

mdl_data.drop(columns=var_list, inplace=True)

mdl_data = mdl_data.dropna()

mdl_data['Delac_1 WTCT3_dif1'].describe(percentiles=[0.05, 0.95])
mdl_data['normal'] = np.where(mdl_data['Delac_1 WTCT3_dif1'].between(-2.498835, 2.305695), 1, 0)
mdl_data.resample('D')['normal'].mean().sort_values(ascending=False)

s1 = '2022-04-03'
s2 = '2022-04-04'
s3 = '2022-04-06'
s4 = '2022-04-10'
s5 = '2022-04-13'
s6 = '2022-04-14'
s7 = '2022-04-18'
s8 = '2022-04-21'
s9 = '2022-04-28'

mdl_data = pd.concat([mdl_data.loc[s1], mdl_data.loc[s2], mdl_data.loc[s3], mdl_data.loc[s4], mdl_data.loc[s5], mdl_data.loc[s6], mdl_data.loc[s7], mdl_data.loc[s8], mdl_data.loc[s9]], axis=0)

mdl_data = mdl_data.dropna()a

mdl_data.to_pickle("data.pkl")

##############################################################

st.set_page_config(
    page_title = 'Real-Time Zone3 Temp Prediction Dashboard',
    layout = 'wide'
)


st.title('Real-Time / Z3 Temperature Prediction Dashboard')

# creating a single-element container
placeholder = st.empty()


##############################################################
# while True:
#
#     with placeholder.container():
#         # create three columns for KPIS
#         kpi_1, kpi_2, kpi_3 = st.columns(3)
#
#         # fill in those three columns with respective metrics or KPIS
#         kpi_1.metric(label='Zone3 Temp / Delac1', value= , delta=, delta_color= )
#
#
#         # create two columns for charts
#         fig_col_1, fig_col_2 = st.columns(2)
#         with fig_col_1:
#             st.markdown("### First Chart")
#             fig = px.
#             st.write(fig)

##############################################################



st.header('st.checkbox')

st.write ('What would you like to order?')

icecream = st.checkbox('Ice cream')
coffee = st.checkbox('Coffee')
cola = st.checkbox('Cola')

if icecream:
     st.write("Great! Here's some more 🍦")

if coffee:
     st.write("Okay, here's some coffee ☕")

if cola:
     st.write("Here you go 🥤")
