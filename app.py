import streamlit as st
import pandas as pd
import Spam_filter_main as sf

df=pd.read_table("SMSSpamCollection")

model,score=sf.train_model(0,df)

st.title("SMS spam classifier")
st.text(f"The precision score of the model is {round(score, 5)}")
msg = st.text_area("Text to analyze")

if st.button('Test message'):
    st.write(f"This is a {sf.test_msg(model,msg)} !")