import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import plotly.express as px
from numpy import set_printoptions
import shap
from PIL import Image
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Telecommunication Churn Prediction
**Model Deployment:   XGBoost**
""")

col1, col2, col3,    col4,   col5 = st.columns(5)
col1.metric("No. of Rows", 3333)
col2.metric("No. of Columns", 19)
col3.metric("No. of Duplicate values", 00)
col4.metric("No. of Null Values", 00)
col5.metric("No. of Missing Values", 00)

st.write('---')

# Loads the Telecommunication Churn Prediction Dataset
tele = pd.read_csv("telecommunications_churn.csv")

if st.button('Show Dataset'):
    st.header('Telecommunication dataset')
    st.write(tele)

st.write('---')

#Visualisation
chart_select = st.sidebar.selectbox(
    label ="Type of chart",
    options=['Scatterplots','Lineplots','Histogram','Boxplot']
)

numeric_columns = list(tele.select_dtypes(['float','int']).columns)

if chart_select == 'Scatterplots':
    st.subheader('Scatterplot Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis',options=numeric_columns)
        plot = px.scatter(data_frame=tele,x=x_values,y=y_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Histogram':
    st.subheader('Histogram Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        plot = px.histogram(data_frame=tele,x=x_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Lineplots':
    st.subheader('Lineplots Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis',options=numeric_columns)
        plot = px.line(tele,x=x_values,y=y_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Boxplot':
    st.subheader('Boxplot Settings')
    try:
        x_values = st.sidebar.selectbox('X axis',options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis',options=numeric_columns)
        plot = px.box(tele,x=x_values,y=y_values)
        st.write(plot)
    except Exception as e:
        print(e)

# Sidebar
# Header of User Input Parameters
st.sidebar.header('User Input Parameters')

def user_input_features():
    account_length = st.sidebar.number_input("Insert account_length")
    voice_mail_messages = st.sidebar.number_input("Insert voice_mail_messages")
    customer_service_calls = st.sidebar.number_input("Insert customer_service_calls")
    international_plan= st.sidebar.number_input("Insert international_plan")
    day_calls = st.sidebar.number_input("Insert day_calls")
    evening_calls = st.sidebar.number_input("Insert evening_calls")
    night_calls = st.sidebar.number_input("Insert night_calls")
    international_calls = st.sidebar.number_input("Insert international_call")
    international_charge = st.sidebar.number_input("Insert international_charge")
    total_charge = st.sidebar.number_input("Insert total_charge")
    data = {'account_length': account_length,
            'voice_mail_messages': voice_mail_messages,
            'customer_service_calls' : customer_service_calls,
            'international_plan' : international_plan,
            'day_calls': day_calls,
            'evening_calls' : evening_calls,
            'night_calls ' : night_calls,
            'international_calls': international_calls,
            'international_charge': international_charge,
            'total_charge' : total_charge}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Print User input parameters
st.subheader('User Input parameters')
st.write(df)
st.write('---')

# Build Model
tele = pd.read_csv("telecommunications_churn.csv")
tele.drop(['voice_mail_plan','day_mins','evening_mins','night_mins','international_mins',
                  'day_charge','evening_charge','night_charge'],inplace=True,axis = 1)
tele = tele.dropna()

tele = pd.read_csv("telecommunications_churn.csv")

x= tele.drop('churn',axis='columns')
y= tele['churn']

array = tele.values
X = array[:,0:10]
Y = array[:,10]

X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.3)

# Creating model
model = XGBClassifier()

# Training model
model.fit(X_train, Y_train)

set_printoptions(precision=2)
prediction = model.predict(df.values)
prediction_proba = model.predict_proba(df.values)

st.subheader('Predicted Result')
image1 = Image.open('Thumbs up.jpg')
image2 = Image.open('Thumbs down.jpg')

if st.button('Predict'):
    if prediction_proba[0][1] > 0.5 :
        st.image(image1)
        st.write('Customer Is Loyal')

    else :
        st.image(image2)
        st.write('Customer Is Not Loyal')

st.write('---')

if st.button('Prediction Probability'):
    st.subheader('Prediction Probability')
    st.write(prediction_proba)

st.write('---')

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
if st.button('Show SHAP Graphs'):
    st.header('Feature Importance')
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X)
    st.pyplot(bbox_inches='tight')
    st.write('---')
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(bbox_inches='tight')   

    