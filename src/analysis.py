import streamlit as st
import plotly.graph_objects as go

color_mapping = {
    "joy": "gold",
    "neutral": "darkgray",
    "surprise": "orange",
    "anger": "orangered",
    "disgust": "green",
    "fear": "darkviolet",
    "sadness": "cornflowerblue",
}

sentiments = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

def analysis():
    st.title('Analysis')
    if 'df' in st.session_state:
        df = st.session_state['df']
        analysis_page(df)
    else:
        st.write('No Data uploaded')

def analysis_page(df):
    st.subheader(f'Nombre de d\'avis: {df.shape[0]}')


    st.write('---')
    st.subheader("Distribution des sentiments")
    plot_sentiments_distribution(df)


def plot_sentiments_distribution(df):
    mean_sentiments = df[sentiments].mean()

    sentiment_df = mean_sentiments.reset_index()
    sentiment_df.columns = ["sentiment", "mean"]

    fig = go.Figure()

    for sentiment in sentiment_df["sentiment"]:
        fig.add_trace(
            go.Bar(
                x=[sentiment],
                y=sentiment_df[sentiment_df["sentiment"] == sentiment]["mean"],
                name=sentiment,
                marker_color=color_mapping[sentiment],
            )
        )

    layout = go.Layout(
        xaxis=dict(
            title="",
            # titlefont=dict(size=18),
            # tickfont=dict(size=23),
        ),
        yaxis=dict(
            title="Valeur Moyenne",
            # titlefont=dict(size=18),
            # tickfont=dict(size=14),
        ),
        autosize=True,
        bargap=0.6,
        margin=dict(t=20)
    )

    fig.update_layout(layout)
    st.plotly_chart(fig)