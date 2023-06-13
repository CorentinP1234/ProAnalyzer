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

sentiments = ["joy", "anger", "disgust", "fear", "surprise", "neutral", "sadness"]


def analysis():
    st.title("Analysis")
    if "df" in st.session_state:
        df = st.session_state["df"]
        analysis_page(df)
    else:
        st.write("No Data uploaded")


def analysis_page(df):
    st.subheader(f"Nombre de d'avis: {df.shape[0]}")

    st.write("---")
    st.subheader("Distribution des sentiments")
    plot_sentiments_distribution(df)

    st.write("---")
    st.subheader("Sentiments par note")

    generate_sentiments(df, sentiments, color_mapping)

    st.write("---")
    highscore_sentiment(df)


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
        # title='Distribution des sentiments',
        title="",
        # title_x=0.7,
        xaxis=dict(
            title="",
        ),
        yaxis=dict(
            title="Valeur Moyenne",
        ),
        autosize=True,
        bargap=0.6,
        margin=dict(t=20),
    )

    fig.update_layout(layout)
    st.plotly_chart(fig)


import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def generate_sentiments(df, sentiments, color_mapping):
    df["date"] = pd.to_datetime(df["date"])

    time_type = st.selectbox("Select Interval", ("Month", "Year"))
    if time_type == "Month":
        df["interval"] = df["date"].dt.to_period("M")
    else:
        df["interval"] = df["date"].dt.to_period("Y")

    unique_intervals = sorted(df["interval"].unique().astype(str))

    default_end_interval = unique_intervals[-1]
    default_start_interval_index = max(0, len(unique_intervals) - 6)
    default_start_interval = unique_intervals[default_start_interval_index]

    col1, col2 = st.columns(2)
    with col1:
        start_interval = st.selectbox(
            f"Select Starting {time_type}",
            unique_intervals,
            index=default_start_interval_index,
        )
    with col2:
        start_index = unique_intervals.index(start_interval)
        end_interval = st.selectbox(
            f"Select Ending {time_type}",
            unique_intervals[start_index:],
            index=len(unique_intervals[start_index:]) - 1,
        )

    start_interval = pd.Period(start_interval, time_type[0])
    end_interval = pd.Period(end_interval, time_type[0])

    df = df[(df["interval"] >= start_interval) & (df["interval"] <= end_interval)]

    grouped_df = df.groupby("interval")[sentiments].mean().reset_index()

    grouped_df["interval"] = grouped_df["interval"].astype(str)

    fig = go.Figure()

    for sentiment in sentiments:
        fig.add_trace(
            go.Bar(
                x=grouped_df["interval"],
                y=grouped_df[sentiment],
                name=sentiment,
                marker_color=color_mapping[sentiment],
            )
        )

    layout = go.Layout(
        title=f"Sentiments by {time_type}",
        xaxis=dict(title=time_type),
        yaxis=dict(title="Valeur Moyenne"),
        autosize=True,
        bargap=0.6,
        margin=dict(t=20),
    )

    fig.update_layout(layout)
    color_mapping_tmp = {
        "joy": "yellow",
        "neutral": "grey",
        "surprise": "orange",
        "sadness": "blue",
        "disgust": "green",
        "anger": "red",
        "fear": "black",
    }

    col1, col2, col3, col4 = st.columns([1, 3, 1, 1])
    with col1:
        st.plotly_chart(fig)
    with col3:
        for sentiment in sentiments[:4]:
            current_value = grouped_df[sentiment].iloc[-1]
            previous_value = (
                grouped_df[sentiment].iloc[-2]
                if len(grouped_df[sentiment]) > 1
                else current_value
            )
            delta = current_value - previous_value
            st.metric(
                label=f"{sentiment.capitalize()}",
                value=f"{current_value:.2f}",
                delta=f"{delta:.2f}",
            )
    with col4:
        for sentiment in sentiments[4:]:
            current_value = grouped_df[sentiment].iloc[-1]
            previous_value = (
                grouped_df[sentiment].iloc[-2]
                if len(grouped_df[sentiment]) > 1
                else current_value
            )
            delta = current_value - previous_value
            st.metric(
                label=f"{sentiment.capitalize()}",
                value=f"{current_value:.2f}",
                delta=f"{delta:.2f}",
            )


# ------------------


def highscore_sentiment(df):
    for sentiment in sentiments:
        # Trier le DataFrame par le score de l'émotion en ordre décroissant
        sorted_df = df.sort_values(by=sentiment, ascending=False)
        # Obtenir la ligne avec le score le plus élevé pour l'émotion
        if sentiment == "anger" or "fear" or "disgust":
            filtered_df = sorted_df["joy"] < 0, 1
        else:
            filtered_df = sorted_df
        top_scores = sorted_df.head(2)
        # Afficher les informations de la ligne
        st.subheader(f"{sentiment}")
        st.write("------------------------")
        for index, row in top_scores.iterrows():
            comment = row["text"]
            st.markdown(f">{comment}")
