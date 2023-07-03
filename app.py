import streamlit as st
from streamlit_option_menu import option_menu
from src.about_us import about_us 
from src.analysis import analysis 
from src.home import home 
from src.how_it_works import how_it_works 
from src.login import login 
from src.signup import signup 

def main():
    st.set_page_config(layout="wide")
    with st.sidebar:
        selected = option_menu(
            menu_title="Pro Analyzer",
            options=["Home", "Analysis", "How it works", "About us"],
            menu_icon='bar-chart-fill',
            icons=['house-fill', 'bar-chart-line-fill', 'question-circle-fill', 'info-circle-fill']
        )

    pages = {
        "Home": home,
        "Analysis": analysis,
        "How it works": how_it_works,
        "About us": about_us,
        "Login": login,
        "Sign Up": signup,
    }

    page = pages[selected]
    page()

if __name__ == "__main__":
    main()
