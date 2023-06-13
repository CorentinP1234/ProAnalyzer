import streamlit as st

color_mapping = {
    "joy": "gold",
    "neutral": "darkgray",
    "surprise": "orange",
    "anger": "orangered",
    "disgust": "green",
    "fear": "darkviolet",
    "sadness": "cornflowerblue",
}

def analysis():
    st.title('Analysis')
    if 'df' in st.session_state:
        df = st.session_state['df']
        st.subheader(f'Nombre de d\'avis: {df.shape[0]}')
    else:
        st.write('No Data uploaded')


def plot1():
    ...