import streamlit as st
import sqlite3


def create_connection():
    conn = sqlite3.connect('users.txt')
    return conn


def create_table(conn):
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users
                      (username TEXT PRIMARY KEY, password TEXT)''')


def signup():
    st.title('Sign Up')
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    confirm_password = st.text_input("Confirm Password", type='password')

    if st.button('Sign Up'):
        if password == confirm_password:
            conn = create_connection()
            create_table(conn)

            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()

            st.success("Sign up successful! You can now log in.")
        else:
            st.error("Password and confirm password do not match.")


if __name__ == "_main_":
    signup()