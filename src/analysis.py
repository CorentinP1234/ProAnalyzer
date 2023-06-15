import streamlit as st
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
import inflect
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

color_mapping_seven = {
    "joy": "gold",
    "neutral": "darkgray",
    "surprise": "orange",
    "anger": "orangered",
    "disgust": "green",
    "fear": "darkviolet",
    "sadness": "cornflowerblue",
}

color_mapping_pos_neg = {
    'neg': 'red',
    'pos': 'green'
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
    st.header(f"{df.shape[0]} reviews to analyse")
    st.write('---')
    n = 3
    number_word = get_number_word(n)
    st.header(f'The {number_word} Most Helpful Reviews')
    print_top_helpful_comments(df, 3)

    st.write("---")
    st.subheader("Distribution des sentiments")
    plot_sentiments_distribution(df)

    st.write("---")
    st.subheader("Sentiments par note")

    generate_sentiments_per_month(df, sentiments, color_mapping_seven, 1)

    st.write("---")
    st.subheader("Sentiments par note")

    generate_sentiments_per_month_min_max(df, ['pos', 'neg'], color_mapping_pos_neg, 4)

    st.write('---')
    display_max_values(df)
    st.write("---")
    highscore_sentiment(df)


# ----
def get_number_word(n):
    p = inflect.engine()
    return p.number_to_words(n).capitalize()

def print_top_helpful_comments(df, n):
    top_comments = df.nlargest(n, 'numHelpful')
    
    for index, row in top_comments.iterrows():
        numHelpful = row['numHelpful']
        helpfulText = f'**{int(numHelpful)}** people found this review helpful'
        title = row['title'].capitalize()
        title = f'##### {title}' 
        text = row['text'].replace('$', ':dollar:')
        text = f"> {text} \n > \n > {helpfulText}"
        st.markdown(title)
        st.write(text)
        

# ----
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
                marker_color=color_mapping_seven[sentiment],
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


# ---

import pandas as pd
import streamlit as st

def display_max_values(df):
    # Get the top 3 texts with the highest 'neg' values
    top_neg_texts = df.nlargest(3, 'neg')['text']

    # Get the top 3 texts with the highest 'pos' values
    top_pos_texts = df.nlargest(3, 'pos')['text']

    # Display the top 'pos' texts
    st.subheader("Top 3 positive texts:")
    for i, text in enumerate(top_pos_texts, 1):
        st.write(f"{i}. {text}")

    # Display the top 'neg' texts
    st.subheader("Top 3 negative texts:")
    for i, text in enumerate(top_neg_texts, 1):
        st.write(f"{i}. {text}")

    



# ---

def generate_sentiments_per_month_min_max(df, sentiments, color_mapping, key):
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
        key=key+1
    )

    start_interval = pd.Period(date1, time_type[0])
    end_interval = pd.Period(date2, time_type[0])

    df = df[(df["interval"] >= start_interval) & (df["interval"] <= end_interval)]

    grouped_df = df.groupby("interval")[sentiments].mean().reset_index()

    # Normalize the sentiment scores with min-max normalization
    for sentiment in sentiments:
        grouped_df[sentiment] = (grouped_df[sentiment] - grouped_df[sentiment].min()) / (grouped_df[sentiment].max() - grouped_df[sentiment].min())

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
        key=key+1
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
