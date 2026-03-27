
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sklearn.metrics import roc_curve, auc

st.set_page_config(layout="wide")
st.title("🚀 Fintech Analytics Studio")

data = pd.read_csv("dataset.csv")

tab1, tab2, tab3, tab4 = st.tabs(["Overview","Descriptive","Predictive","Future Scoring"])

# Overview
with tab1:
    st.metric("Rows", len(data))
    st.metric("Columns", len(data.columns))
    st.metric("Interested %", round(data['interest'].mean()*100,2))

# Descriptive
with tab2:
    st.plotly_chart(px.histogram(data,x="revenue"))
    st.plotly_chart(px.histogram(data,x="payment_delay"))
    st.plotly_chart(px.histogram(data,x="credit_need"))
    st.plotly_chart(px.pie(data,names="city_tier"))

# Predictive
with tab3:
    st.write("Model Performance")
    st.write({'accuracy': 0.988, 'precision': 0.9657142857142857, 'recall': 1.0, 'f1': 0.9825581395348837})

    clf = pickle.load(open("clf.pkl","rb"))
    X = pd.read_csv("dataset_encoded.csv").drop(["interest","loan_size"], axis=1)
    y = pd.read_csv("dataset_encoded.csv")["interest"]

    y_prob = clf.predict_proba(X)[:,1]
    fpr,tpr,_ = roc_curve(y,y_prob)
    roc_auc = auc(fpr,tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr,y=tpr,name="ROC"))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],name="Random"))
    st.plotly_chart(fig)

# Future scoring
with tab4:
    clf = pickle.load(open("clf.pkl","rb"))
    reg = pickle.load(open("reg.pkl","rb"))

    uploaded = st.file_uploader("Upload new data")

    if uploaded:
        df_new = pd.read_csv(uploaded)
        df_new["probability"] = clf.predict_proba(df_new)[:,1]
        df_new["loan"] = reg.predict(df_new)
        st.write(df_new)
