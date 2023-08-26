import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import warnings
warnings.filterwarnings('ignore')


# In[1]:


import pandas as pd
import numpy as np
import math
import datetime as dt

import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from plotly.offline import plot, iplot, init_notebook_mode


# In[3]:


df=pd.read_csv(".\Dataset\Gemini_BTCUSD_1h.csv",skiprows=1)


# In[5]:


df=df.iloc[::-1].reset_index()


# In[7]:


df=df[['date','close']]


# In[11]:


df_close=df.copy()


# In[12]:


del df_close['date']


# In[14]:


scaler=MinMaxScaler(feature_range=(0,1))
df_close=scaler.fit_transform(np.array(df_close).reshape(-1,1))
print(df_close.shape)


# In[15]:


training_size=int(len(df_close)*0.75)
test_size=len(df_close)-training_size


train_data,test_data=df_close[0:training_size,:],df_close[training_size:len(df_close),:1]


print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)


# In[17]:


# Prepare train data for time series analysis
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 360
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)

# In[19]:


# # Biulding Model
# my_model = XGBRegressor(n_estimators=200)
# my_model.fit(X_train, y_train, verbose=False)


# In[20]:


import pickle
# filename = 'xgboost.pkl'
# pickle.dump(my_model, open(filename, 'wb'))


# In[21]:


# del my_model

# my_model


# In[22]:


file = open('xgboost.pkl', 'rb')

# dump information to that file
my_model = pickle.load(file)


# In[23]:


my_model


# In[24]:


predictions = my_model.predict(X_test)
print("Mean Absolute Error - MAE : " + str(mean_absolute_error(y_test, predictions)))
print("Root Mean squared Error - RMSE : " + str(math.sqrt(mean_squared_error(y_test, predictions))))


# In[25]:


train_predict=my_model.predict(X_train)
test_predict=my_model.predict(X_test)

train_predict = train_predict.reshape(-1,1)
test_predict = test_predict.reshape(-1,1)

print("Train data prediction:", train_predict.shape)
print("Test data prediction:", test_predict.shape)


# In[26]:

# Transform back to original form

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))


# In[27]:


# shift train predictions for plotting

look_back=time_step
trainPredictPlot = np.empty_like(df_close)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
print("Train predicted data: ", trainPredictPlot.shape)


# In[28]:


# shift test predictions for plotting
testPredictPlot = np.empty_like(df_close)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df_close)-1, :] = test_predict
print("Test predicted data: ", testPredictPlot.shape)



# In[29]:


print("Train data R2 score:", r2_score(original_ytrain, train_predict))
print("Test data R2 score:", r2_score(original_ytest, test_predict))


# In[30]:


names = cycle(['Original close price','Train predicted close price','Test predicted close price'])

plotdf = pd.DataFrame({'date': df['date'],
                       'original_close': df['close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})
new_new  = pd.DataFrame({
                       'original_close': df['close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})
# st.dataframe(plotdf)
# fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
#                                           plotdf['test_predicted_close']],
#               labels={'value':'Close price','date': 'Date'})
# fig.update_layout(title_text='Comparision between original close price vs predicted close price',
#                   plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
# fig.for_each_trace(lambda t:  t.update(name = next(names)))
#
# fig.update_xaxes(showgrid=False)
# fig.update_yaxes(showgrid=False)
# fig.show()


st.line_chart(new_new)

# In[31]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 10*24
while(i<pred_days):

    if(len(temp_input)>time_step):

        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)

        yhat = my_model.predict(x_input)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat.tolist())
        temp_input=temp_input[1:]

        lst_output.extend(yhat.tolist())
        i=i+1

    else:
        yhat = my_model.predict(x_input)

        temp_input.extend(yhat.tolist())
        lst_output.extend(yhat.tolist())

        i=i+1

print("Output of predicted next days: ", len(lst_output))


# In[ ]:





# In[32]:


last_days=np.arange(1,time_step+1)
day_pred=np.arange(time_step+1,time_step+pred_days+1)


# In[33]:


temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = scaler.inverse_transform(df_close[len(df_close)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

new_pred_plot = pd.DataFrame({
    'last_original_days_value':last_original_days_value,
    'next_predicted_days_value':next_predicted_days_value
})

# names = cycle(['Last 15 days close price','Predicted next 10 days close price'])
#
# fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
#                                                       new_pred_plot['next_predicted_days_value']],
#               labels={'value': 'Close price','index': 'Timestamp'})
# fig.update_layout(title_text='Compare last 15 days vs next 10 days',
#                   plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
# fig.for_each_trace(lambda t:  t.update(name = next(names)))
# fig.update_xaxes(showgrid=False)
# fig.update_yaxes(showgrid=False)
# fig.show()
# st.pyplot(fig)

st.dataframe(new_pred_plot)
print("end here")
st.line_chart(new_pred_plot)

st.write()


