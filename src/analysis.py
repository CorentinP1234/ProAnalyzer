import re
import sys
import pandas as pd
import streamlit as st
from streamlit_tags import st_tags
from annotated_text import annotation
from transformers import pipeline
import asyncio
import torch
import subprocess
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.plots import (
    plot_sentiments_distribution, 
    plot_sentiments_per_month_min_max, 
    generate_sentiments_per_month
)
# ----------
def bigrames(df):
    data = df['text']
    data = data.fillna('')
    stop_words = stopwords.words('english')  # Utiliser stopwords.words('english') pour obtenir les stopwords
    stop_words.extend(["product", "amazon", "fire", "kindle"])  # Ajouter les mots à la liste des stopwords

    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(2, 2))
    X = vectorizer.fit_transform(data)

    feature_names = vectorizer.get_feature_names_out()
    top_words = []

    for idx, word in enumerate(feature_names):
        score = X.getcol(idx).sum()
        top_words.append((word, score))

    top_words.sort(key=lambda x: x[1], reverse=True)
    mots_pondérés = {word: score for word, score in top_words[:50]}
    return mots_pondérés

def monogramme(df):
    df = pd.read_csv("comparatif_Fire Tablet, 7 Display, Wi-Fi, 8 GB.csv", low_memory=False)
    print("check")
    data = df[df['vaders_pos'] >= 0.25]['text']
    data.fillna('', inplace=True)

    stop_words = stopwords.words('english')  # Utiliser stopwords.words('english') pour obtenir les stopwords
    stop_words.extend(["bought", "product", "amazon", "fire", "kindle", "echo", "alexa", "great", "use","good","love", "easy", "loves", "fast", "device", "best", "nice", "also", "everything"])  # Ajouter les mots à la liste des stopwords

    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 1))
    X = vectorizer.fit_transform(data)

    feature_names = vectorizer.get_feature_names_out()
    top_words = []

    for idx, word in enumerate(feature_names):
        score = X.getcol(idx).sum()
        top_words.append((word, score))

    top_words.sort(key=lambda x: x[1], reverse=True)

    for word, score in top_words[:25]:
        st.write(word, score)

def generate_wordcloud(mots):
    st.title("Nuage de mots")
    wordcloud = WordCloud(width=800, height=400, background_color='black', relative_scaling=0.5, max_words=50, prefer_horizontal=0.7).generate_from_frequencies(mots)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
# -------------

COLORS = {"blue": "#0000FF", "red": "#FF4B4B", "green": "#31ab2b"}
EMOTIONS = ["joy", "anger", "disgust", "fear", "surprise", "neutral", "sadness"]
COLOR_MAPPING_EMOTION = {
    "joy": "gold",
    "neutral": "darkgray",
    "surprise": "orange",
    "anger": "orangered",
    "disgust": "green",
    "fear": "darkviolet",
    "sadness": "cornflowerblue",
}
COLOR_MAPPING_SENTIMENTS = {"negative": "red", "positive": "green"}

def get_comments_with_keywords(df, keywords):
    for keyword in keywords:
        df = df[df["text"].str.contains(r"\b" + re.escape(keyword) + r"\b", case=False)]
    return df

def get_df_filtered(df, key):
    keywords_ = st_tags(
        label="Enter Keywords:",
        text="Press enter to add more",
        value=[],
        suggestions=["Price", "Product", "Sales"],
        maxtags=4,
        key=key,
    )
    keywords = [word.lower() for word in keywords_]
    df = get_comments_with_keywords(df, keywords)
    st.write(f"{len(df)} comments found containing all the keywords." if df.empty else "There are no comments containing all the keywords.")
    return df, keywords

def write_review(title, text, rating=1, text_font_size=15):
    stars = "★" * rating + "☆" * (5 - rating)
    review_html = f"""
    <div style="background: linear-gradient(to right, #262730, #1f1c28); 
                border-radius: 7px; 
                padding: 2px 15px 5px 15px; 
                border: 1px solid #555555; 
                box-shadow: 0 4px 6px 0 hsla(0, 0%, 0%, 0.2);">
        <div style="font-size: 25px; color: #FFFFFF;">
            <span style="font-size: 20px; color: gold; padding-right: 10px;">
                <strong>{stars}</strong>
            </span>
            <strong>{title}</strong>
        </div>
        <div style="margin-top: 5px;"></div>
        <div style="font-size: {text_font_size}px; font-family: Arial, sans-serif; color: #FFFFFF;">"{text}"</div>
    </div>
    """
    st.markdown(review_html, unsafe_allow_html=True)

def write_review_w_keywords(title, text, rating, keywords, color):
    keywords_lower = [keyword.lower() for keyword in keywords]
    for keyword in keywords_lower:
        if keyword in text.lower():
            text = text.replace(keyword, str(annotation(keyword, "", color)))
    write_review(title, text, rating)


def display_top_helpful_comments(df, n, keywords):
    st.header(f"Top {n} helpful reviews according to the users")
    for index, row in df.nlargest(n, "numHelpful").iterrows():
        title, text, rating = row["title"].capitalize(), row["text"], int(row["rating"])
        write_review_w_keywords(title, text, rating, keywords, COLORS["blue"])
        st.write(f":green[{int(row['numHelpful'])}] people found this review helpful")

def display_top_pos_neg_reviews(df, n, keywords):
    display_reviews(df.nlargest(n, "positive"), keywords, "positive", "green")
    display_reviews(df.nlargest(n, "negative"), keywords, "negative", "red")

def display_reviews(df, keywords, sentiment, color):
    st.header(f"Top {len(df)} {sentiment} reviews")
    for index, row in df.iterrows():
        title, text, rating = row["title"].capitalize(), row["text"], int(row["rating"])
        write_review_w_keywords(title, text, rating, keywords, COLORS[color])
        st.write(f"This review got a :{color}[{round(row[sentiment], 3)}] {sentiment} score.")

def model(df):
    st.write(torch.cuda.is_available())
    model_name = 'j-hartmann/emotion-english-distilroberta-base'

    classifier = pipeline(
        "text-classification",
        model=model_name,
        truncation=True,
        top_k=None,
        device=0
    )
    sentiment_scores = classifier((df['title']+ " " + df['text']).tolist())
    st.write(sentiment_scores[:3])


def display_emotions_sentiments_analysis(df):
    st.title("Emotions Analysis")
    st.header("Emotions distribution")
    plot_sentiments_distribution(df, EMOTIONS, COLOR_MAPPING_EMOTION)
    st.write("---")
    st.header("Emotions per rating")
    generate_sentiments_per_month(df, EMOTIONS, COLOR_MAPPING_EMOTION, 1)
    st.write("---")
    st.title("Sentiments Analysis")
    st.write("---")
    st.header("Sentiments distribution")
    plot_sentiments_distribution(df, ["positive", "negative"], COLOR_MAPPING_SENTIMENTS)
    st.write("---")
    st.header("Sentiments per rating")
    plot_sentiments_per_month_min_max(df, ["positive", "negative"], COLOR_MAPPING_SENTIMENTS, 40)



def analysis_page(df):
    st.subheader(st.session_state["name"])
    st.subheader(f"{df.shape[0]} reviews to analyse")
    # model(df)
    # results = subprocess.run([f"{sys.executable}", "src/model.py"], capture_output=True, text=True)
    # st.write(results.stdout)
    
    st.write("---")

    df_filtered, keywords = get_df_filtered(df, 3)
    n = st.number_input("Number of helpful reviews to show", min_value=1, max_value=15, value=3)
    st.write("---")

    display_top_helpful_comments(df_filtered, n, keywords)
    st.write("---")
    display_top_pos_neg_reviews(df_filtered, n, keywords)
    st.write("---")
    display_emotions_sentiments_analysis(df)
    st.write('---')
    highscore_sentiment(df)
    st.write('---')
    mots = bigrames(df)
    generate_wordcloud(mots)

def analysis():
    st.title("Analysis")
    if "df" in st.session_state:
        df = st.session_state["df"]
        df = df.rename(columns={'pos': 'positive', 'neg': 'negative'})
        analysis_page(df)
    else:
        st.write("No Data uploaded")

def highscore_sentiment(df):
    for sentiment in EMOTIONS:
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