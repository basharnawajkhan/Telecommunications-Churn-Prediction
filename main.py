import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

st.title("Telecommunications Churn Prediction")
st.sidebar.header("User Input Parameters")

def user_input_features():
    voice_mail_plan = st.sidebar.selectbox("Voice Mail Plan", ("1", "0"))
    voice_mail_messages	 = st.sidebar.number_input("Number voice mail messages")
    international_mins = st.sidebar.number_input("Total international mins")
    customer_service_calls = st.sidebar.number_input("Number customer service calls")
    international_plan = st.sidebar.selectbox("International Plan", ("1", "0"))
    international_calls = st.sidebar.number_input("Number international calls")
    international_charge = st.sidebar.number_input("Total international charge")
    total_charge = st.sidebar.number_input("Total charge")
    Data = {"voice_mail_plan":voice_mail_plan, "voice_mail_messages":voice_mail_messages, "international_mins":international_mins,
           "customer_service_calls":customer_service_calls, "international_plan":international_plan,
           "international_calls":international_calls, "international_charge":international_charge, "total_charge":total_charge}
    features = pd.DataFrame(Data, index=[0])
    return features

result = user_input_features()

st.subheader("User Input Parameters")
st.write(result)

data = pd.read_csv("cleaned_data.csv")

final_df = data.drop(["Unnamed: 0"], axis=1)

from sklearn.model_selection import train_test_split

X = final_df.iloc[:,:-1]
Y = final_df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, shuffle=True)

def norm_func (i):
    x = (i - i.min())/ (i.max()-i.min())
    return (x)

x_train = norm_func(x_train)
x_test =  norm_func(x_test)

from sklearn.svm import SVC
Model = SVC(kernel="rbf", random_state=7, gamma=100)
Model.fit(x_train, y_train)
Prediction = Model.predict(result)

st.subheader("Predicted Result")
st.write(Prediction)