import streamlit as st

def login():
    st.title('Login to Your Account')
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button('Login'):
        st.write("Implement login functionality here.")
