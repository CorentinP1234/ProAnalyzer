import streamlit as st
import pandas as pd



import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def home():
    st.title('Welcome to Our Application')
    st.write("This is the home page of our application.")
    get_uploaded_file()

def get_uploaded_file():
    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df