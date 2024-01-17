import streamlit as st
import pandas as pd
import Spam_filter_main as sf

st.set_page_config(page_title="SMS spam filter")

df=pd.read_table("SMSSpamCollection")

st.title("SMS spam classifier")

trained=0
whole="NaN"

if st.button('Train with the whole dataset'):
    model,score=sf.train_model(0,1,df)
    whole=1
    trained=1
if st.button('Train with 70% of the dataset'):
    model,score=sf.train_model(0,0,df)
    trained=1
    whole=0

if whole==1 : st.text(f"The training accuracy of the model is {round(score, 5)}")
elif whole==0 : st.text(f"The testing accuracy of the model is {round(score, 5)}")

msg = st.text_area("Text to analyze")

if trained:
    if st.button('Test message'):
        st.write(f"This is a {sf.test_msg(model,msg)} !")
else : st.button('Test message',disabled=True)