import streamlit as st

def signup():
    st.title('Sign Up')
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    confirm_password = st.text_input("Confirm Password", type='password')
    if st.button('Sign Up'):
        st.write("Implement sign up functionality here.")
