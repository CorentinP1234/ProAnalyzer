import streamlit as st
import sqlite3
from src.analysis import analysis


def create_connection():
    conn = sqlite3.connect('users.txt')
    return conn




def login():

    ph1 = st.empty()
    ph3 = st.empty()
    ph4 = st.empty()
    ph5 = st.empty()

    conn = create_connection()

    ph1.title('Login to Your Account')
    username = ph3.text_input("Username")
    password = ph4.text_input("Password", type='password')

    if ph5.button('Login'):

        ph1.empty()
        ph3.empty()
        ph4.empty()
        ph5.empty()
        st.session_state.runpage = analysis

        st.session_state["selected"]="Analysis"
        st.session_state.runpage()




def verify_credentials(conn, username, password):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()

    if result and result[1] == password:
        return True
    else:
        return False


if __name__ == "_main_":
    login()
