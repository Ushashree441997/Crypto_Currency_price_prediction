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
option = st.selectbox(
    'Select the type of the coin',
    ('BTCUSD', 'ETHUSD', 'DOGEUSD', 'LTCUSD'))

st.write('You selected:', option)
# ---------------------------------------------------------------
df = pd.read_csv("E:\MumBaiKharGhar\Major Project DBDA\Project\MajorProject\Gemini_BTCUSD_d.csv",skiprows=1)

col1,col2 = st.columns((2))
df["date"] = pd.to_datetime(df["date"])
# #Getting min and max date
startDate = pd.to_datetime(df["date"]).min()
endDate = pd.to_datetime(df["date"]).max()

with col1:
    date1 = pd.to_datetime(st.date_input("From : ",startDate))
with col2:
    date2 = pd.to_datetime(st.date_input("To : ",endDate))

#-------------------------------------------------------------------
if st.button('Predict'):
    btn_col1,btn_col2 = st.columns((2))
    with btn_col1:
        st.write('Graph and prediction for the particular bitcoin')
        col1, col2 = st.columns([4,4])
        data = np.random.randn(5, 5)

        col1.subheader("Gemini Bitcoin Coin ")
        col1.line_chart(data=df, x='date', y='close', width=100, height=400)

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
else:
    st.write('Something went wrong!')

