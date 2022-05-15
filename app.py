import streamlit as st

st.set_page_config(
    page_title = 'Reduction Black Dross / Zone3 Temp Prediction Dashboard',
    layout = 'wide')

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import datetime
import pickle
from io import BytesIO

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)


def df_filter(message,df):

        slider_1, slider_2 = st.slider('%s' % (message), 0, len(df)-1, [0,len(df)-1], 1)

        while len(str(df.iloc[slider_1][1]).replace('.0','')) < 4:
            df.iloc[slider_1,1] = '0' + str(df.iloc[slider_1][1]).replace('.0','')

        while len(str(df.iloc[slider_2][1]).replace('.0','')) < 4:
            df.iloc[slider_2,1] = '0' + str(df.iloc[slider_1][1]).replace('.0','')

        start_date = datetime.datetime.strptime(str(df.iloc[slider_1][0]).replace('.0','') + str(df.iloc[slider_1][1]).replace('.0',''),'%Y%m%d%H%M%S')
        start_date = start_date.strftime('%d %b %Y, %I:%M')

        end_date = datetime.datetime.strptime(str(df.iloc[slider_2][0]).replace('.0','') + str(df.iloc[slider_2][1]).replace('.0',''),'%Y%m%d%H%M%S')
        end_date = end_date.strftime('%d %b %Y, %I:%M')

        st.info('Start: **%s** **~** End: **%s**' % (start_date,end_date))

        filtered_df = df.iloc[slider_1:slider_2+1][:].reset_index(drop=True)

        return filtered_df

if __name__ == '__main__':

    # Sidebar
    with st.sidebar:
        viewer = st.radio(
            "Choose a Viewer",
            ['Plant Engineer', 'Data Science'])

    ## Start of the page
    st.title('Real-Time / Z3&4 Temperature Prediction Dashboard')

    st.text(" ")
    st.text(" ")

    st.markdown('### Datetime Filter')

    st.text(" ")
    st.text(" ")

    PI_df = pd.read_pickle('PI_df.pkl')
    PI_df.reset_index(inplace=True)
    PI_df = PI_df.rename(columns = {'index':'datetime'})
    PI_df['date-time'] = PI_df['datetime'].apply(lambda x : x.strftime('%Y%m%d-%H%M'))
    PI_df[['date', 'time']] = PI_df['date-time'].str.split("-", 1, expand=True)

    PI_df = PI_df[['date', 'time', 'Delac_1 WTCT3', 'Delac_2 WTCT3', 'Delac_1 WTCT4', 'Delac_2 WTCT4', 'Delac_1 Kiln_Temperature_Zone_3_Prediction', 'Delac_2 Kiln_Temperature_Zone_3_Prediction', 'Delac_1 Kiln_Temperature_Zone_4_Prediction', 'Delac_2 Kiln_Temperature_Zone_4_Prediction']]
    filtered_PI_df = df_filter('Move sliders to filter dataframe', PI_df)

    st.text(" ")



    if viewer == 'Plant Engineer':
        # Delac_1_z3_standard_deviation_chart:
        fig = plt.figure(figsize=(13, 10))
        sns.distplot(filtered_PI_df['Delac_1 WTCT3'], color='blue', label='Before', kde=True, hist=False)
        sns.distplot(filtered_PI_df['Delac_2 WTCT3'], color='red', label='After', kde=True, hist=False)
        plt.legend()
        plt.title('Delac 1 z3 Distplot', fontsize=20)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)

        # Delac_1 z4 Standard deviation
        fig = plt.figure(figsize=(13, 10))
        sns.distplot(filtered_PI_df['Delac_1 WTCT3'], color='blue', label='Before', kde=True, hist=False)
        sns.distplot(filtered_PI_df['Delac_2 WTCT3'], color='red', label='After', kde=True, hist=False)
        plt.legend()
        plt.title('Delac 1 z4 Distplot', fontsize=20)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)

        # Delac_2 z3 Standard deviation
        fig = plt.figure(figsize=(13, 10))
        sns.distplot(filtered_PI_df['Delac_2 WTCT3'], color='blue', label='Before', kde=True, hist=False)
        sns.distplot(filtered_PI_df['Delac_1 WTCT3'], color='red', label='After', kde=True, hist=False)
        plt.legend()
        plt.title('Delac 2 z3 Distplot', fontsize=20)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)

        # Delac_2 z4 Standard deviation
        fig = plt.figure(figsize=(13, 10))
        sns.distplot(filtered_PI_df['Delac_2 WTCT4'], color='blue', label='Before', kde=True, hist=False)
        sns.distplot(filtered_PI_df['Delac_1 WTCT4'], color='red', label='After', kde=True, hist=False)
        plt.legend()
        plt.title('Delac 2 z4 Distplot', fontsize=20)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)

    else:

        # Delac_1_Z3 True vs Predic
        fig = plt.figure(figsize=(13, 10))
        plt.plot(filtered_PI_df.index, filtered_PI_df['Delac_1 WTCT3'], color='blue')
        plt.plot(filtered_PI_df.index, filtered_PI_df['Delac_1 Kiln_Temperature_Zone_3_Prediction'], color='red')
        plt.legend(['True', 'Pred'], loc='upper right')
        plt.title('Delac1 Z3 True vs Predict', fontsize=20)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)

        # Delac_1_Z4 True vs Predict
        fig = plt.figure(figsize=(13, 10))
        plt.plot(filtered_PI_df.index, filtered_PI_df['Delac_1 WTCT4'], color='blue')
        plt.plot(filtered_PI_df.index, filtered_PI_df['Delac_1 Kiln_Temperature_Zone_4_Prediction'], color='red')
        plt.legend(['True', 'Pred'], loc='upper right')
        plt.title('Delac 1 Z4 True vs Predict', fontsize=20)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)

        # Delac_2_Z3 True vs Predict
        fig = plt.figure(figsize=(13, 10))
        plt.plot(filtered_PI_df.index, filtered_PI_df['Delac_2 WTCT3'], color='blue')
        plt.plot(filtered_PI_df.index, filtered_PI_df['Delac_2 Kiln_Temperature_Zone_3_Prediction'], color='red')
        plt.legend(['True', 'Pred'], loc='upper right')
        plt.title('Delac 2 Z3 True vs Predict', fontsize=20)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)

        # Delac_2_Z4 True vs Predict
        fig = plt.figure(figsize=(13, 10))
        plt.plot(filtered_PI_df.index, filtered_PI_df['Delac_2 WTCT4'], color='blue')
        plt.plot(filtered_PI_df.index, filtered_PI_df['Delac_2 Kiln_Temperature_Zone_4_Prediction'], color='red')
        plt.legend(['True', 'Pred'], loc='upper right')
        plt.title('Delac 2 Z4 True vs Predict', fontsize=20)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)
