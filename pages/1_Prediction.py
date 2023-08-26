import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(page_title="Crypto Currency Prediction",page_icon = ":chart_with_upwards_trend",layout="wide")
st.title(" :chart_with_upwards_trend: Crypto Currency Prediction")

st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

# browsering the files ----------------------------------------------
# f1 = st.file_uploader(":file folder: Upload a file",type = (["csv","xlsx","txt","xls"]))
# if f1 is not None:
#     filename = f1.name
#     st.write(filename)
#     df = pd.read_csv(filename)
# else:
#     os.chdir(r"E:\MumBaiKharGhar\Major Project DBDA\Project\MajorProject")
#     df = pd.read_csv("Gemini_BTCUSD_d.csv",skiprows=1)
# ---------------------------------------------------------------------------------------

# --------------------------------------------------------------

col1,col2,col3 = st.columns(3)
with col1:
    option = st.selectbox(
    'Select the type of the coin',
    ('BTCUSD', 'ETHUSD', 'DOGEUSD', 'LTCUSD'))
with col2:
    option2 = st.selectbox(
        'Select the type of the coin',
        ('LSTM', 'XGBoost', 'ARIMA'))

with col3:
    date2 = pd.to_datetime(st.date_input("Date ", min_value=datetime.datetime.now()))

st.write(date2)
st.write(date2 - datetime.datetime.now())

if st.button("Predict"):

    st.write('You selected:', option)
# ---------------------------------------------------------------
    df = pd.read_csv(".\Dataset\Gemini_BTCUSD_1h.csv",skiprows=1)

    btn_col1,btn_col2 = st.columns((2))
    with btn_col1:
        st.write('Graph and prediction for the particular bitcoin')
        st.subheader("Gemini Bitcoin Coin ")
        st.line_chart(data=df, x='date', y='close', width=300, height=400)

    with btn_col2:
        df = pd.DataFrame(
            {
                "name": ["Accuracy", "Precision", "Recall", "F1 Score", "RMSE", "MES"],
                "Values": [9.6757, 7.6567, 5.552, 9.6757, 7.6567, 5.552]
            }
        )
        st.dataframe(
            df,
            column_config={
                "name": "Prediction",
                "Values": "Values"
            },
            hide_index=True,
        )
        # Xgboost()
else:
    st.write('Something went wrong!')

