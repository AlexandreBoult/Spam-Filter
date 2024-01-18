import streamlit as st
import pandas as pd
import Spam_filter_main as sf

st.set_page_config(page_title="SMS spam filter")

df=pd.read_table("SMSSpamCollection")

st.title("SMS spam classifier")

if 'button_test_disabled' not in st.session_state:
    st.session_state.button_test_disabled = True
    st.session_state.which = 0
    

whole="NaN"

if st.button('Train with the whole dataset'):
    st.session_state.which = 100
    st.session_state.model=sf.train_model(0,1,df)

if st.button('Train with 70% of the dataset'):
    st.session_state.which = 70
    st.session_state.model=sf.train_model(0,0,df)

if st.session_state.which==100 :
    st.session_state.button_test_disabled = False
    model,score=st.session_state.model
    st.text(f"The training accuracy of the model is {round(score, 5)}")

elif st.session_state.which==70 :
    st.session_state.button_test_disabled = False
    model,score=st.session_state.model
    st.text(f"The testing accuracy of the model is {round(score, 5)}")

msg = st.text_area("Text to analyze")


columns = st.columns(2)
button_test=columns[0].button('Test message',disabled=st.session_state.button_test_disabled)

if button_test:
    columns[1].write(f"This is a {sf.test_msg(model,msg)} !")