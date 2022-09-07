import numpy as np
import pandas as pd
import streamlit as st
import pickle

classifier = pickle.load(open("trained_model.pkl", "rb"))

def norm_func (i):
    x = (i - i.min())/ (i.max()-i.min())
    return (x)

def churn_prediction(input_data):
    array=np.asarray(input_data)
    reshape = array.reshape(1,-1)
    norm = norm_func(reshape)

    prediction = classifier.predict(norm)
    print(prediction)

    if (prediction[0]==0):
        return "The customer is loyal."
    else:
        return "The customer is churn."

def main():

    st.title("Telecommunications Churn Prediction")

    voice_mail_plan = st.sidebar.selectbox("Voice Mail Plan", ("1", "0"))
    voice_mail_messages = st.sidebar.number_input("Number voice mail messages")
    international_mins = st.sidebar.number_input("Total international mins")
    customer_service_calls = st.sidebar.number_input("Number customer service calls")
    international_plan = st.sidebar.selectbox("International Plan", ("1", "0"))
    international_calls = st.sidebar.number_input("Number international calls")
    international_charge = st.sidebar.number_input("Total international charge")
    total_charge = st.sidebar.number_input("Total charge")

    customer_churn = ""

    if st.button("Predict"):
        customer_churn = churn_prediction([voice_mail_plan, voice_mail_messages, international_mins, customer_service_calls, international_plan, international_calls, international_charge, total_charge])

    st.success(customer_churn)


if __name__ == "__main__":
    main()

