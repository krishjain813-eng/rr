
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

st.set_page_config(layout="wide")
st.title("📊 Professional Fintech Dashboard")

data = pd.read_csv("dataset.csv")

st.subheader("Overview")
st.dataframe(data.head())

# charts
st.plotly_chart(px.histogram(data, x="Revenue"))
st.plotly_chart(px.pie(data, names="Location"))
st.plotly_chart(px.bar(data["Business_Type"].value_counts().reset_index(), x="index", y="Business_Type"))
st.plotly_chart(px.histogram(data, x="Credit_Need", color="Business_Type"))
st.plotly_chart(px.histogram(data, x="Payment_Delay"))
st.plotly_chart(px.density_heatmap(data, x="Inventory_Value", y="Revenue"))
st.plotly_chart(px.histogram(data, x="Cash_Shortage"))
st.plotly_chart(px.histogram(data, x="Age_Group"))

# prediction
clf = pickle.load(open("clf.pkl","rb"))
reg = pickle.load(open("reg.pkl","rb"))

uploaded = st.file_uploader("Upload encoded data")

if uploaded:
    df_new = pd.read_csv(uploaded)
    df_new["Prediction"] = clf.predict(df_new)
    df_new["Probability"] = clf.predict_proba(df_new)[:,1]
    df_new["Loan"] = reg.predict(df_new)
    st.write(df_new)
