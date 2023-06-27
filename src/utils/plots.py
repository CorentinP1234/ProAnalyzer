import pandas as pd
from dateutil.relativedelta import relativedelta
import streamlit as st
import plotly.graph_objects as go


def plot_sentiments_per_month_min_max(df, sentiments, color_mapping, key):
    df["date"] = pd.to_datetime(df["date"])

    time_type = st.selectbox("Select Interval", ("Month", "Year"), key=key)
    if time_type == "Month":
        df["interval"] = df["date"].dt.to_period("M")
    else:
        df["interval"] = df["date"].dt.to_period("Y")

    unique_intervals = sorted(df["interval"].unique().astype(str))
    start_date = pd.to_datetime(unique_intervals[0]).to_pydatetime()
    end_date = pd.to_datetime(unique_intervals[-1]).to_pydatetime()
    five_months_prior_end_date = end_date - relativedelta(months=5)

    date1, date2 = st.slider(
        "Schedule your appointment:",
        min_value=start_date,
        max_value=end_date,
        value=(five_months_prior_end_date, end_date),
        key=key + 1,
    )

    start_interval = pd.Period(date1, time_type[0])
    end_interval = pd.Period(date2, time_type[0])

    df = df[(df["interval"] >= start_interval) & (df["interval"] <= end_interval)]

    grouped_df = df.groupby("interval")[sentiments].mean().reset_index()

    # Normalize the sentiment scores with min-max normalization
    for sentiment in sentiments:
        grouped_df[sentiment] = (
            grouped_df[sentiment] - grouped_df[sentiment].min()
        ) / (grouped_df[sentiment].max() - grouped_df[sentiment].min())

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
        yaxis=dict(title="Trends"),
        autosize=True,
        bargap=0.6,
        margin=dict(t=20),
    )

    fig.update_layout(layout)
    st.plotly_chart(fig)


def plot_sentiments_distribution(df, sentiments, color_mapping):
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


def generate_sentiments_per_month(df, sentiments, color_mapping, key):
    df["date"] = pd.to_datetime(df["date"])

    time_type = st.selectbox("Select Interval", ("Month", "Year"), key=key)
    if time_type == "Month":
        df["interval"] = df["date"].dt.to_period("M")
    else:
        df["interval"] = df["date"].dt.to_period("Y")

    unique_intervals = sorted(df["interval"].unique().astype(str))
    start_date = pd.to_datetime(unique_intervals[0]).to_pydatetime()
    end_date = pd.to_datetime(unique_intervals[-1]).to_pydatetime()
    five_months_prior_end_date = end_date - relativedelta(months=5)

    date1, date2 = st.slider(
        "Schedule your appointment:",
        min_value=start_date,
        max_value=end_date,
        value=(five_months_prior_end_date, end_date),
        key=key + 1,
    )

    start_interval = pd.Period(date1, time_type[0])
    end_interval = pd.Period(date2, time_type[0])

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
    st.plotly_chart(fig)