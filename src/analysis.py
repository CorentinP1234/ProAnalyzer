import re
import sys

import nltk
import pandas as pd
import streamlit as st
from streamlit_tags import st_tags
from annotated_text import annotation
from transformers import pipeline
import asyncio
import torch
import numpy as np
import subprocess
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.plots import (
    plot_sentiments_distribution, 
    plot_sentiments_per_month_min_max, 
    generate_sentiments_per_month
)


nltk.download('stopwords')

# def monogramme():
#     df = pd.read_csv("comparatif_Fire Tablet, 7 Display, Wi-Fi, 8 GB.csv", low_memory=False)
#     st.write('check')
#     data = df[df['roberta_neg'] >= 0.17]['text']
#     data.fillna('', inplace=True)

#     stop_words = stopwords.words('english')
#     stop_words.extend(["product", "amazon", "fire", "kindle", "echo", "alexa", "great","easy", "loves", "device", "best", "nice", "also", "everything", "would", "really", "much","excellent","one", "two",'awesome',"first", "friday"])

#     vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 1))
#     X = vectorizer.fit_transform(data)
#     feature_names = vectorizer.get_feature_names_out()
    

#     top_words = []
#     for idx, word in enumerate(feature_names):
#         score = X.getcol(idx).sum()
#         top_words.append((word, score))
#     top_words.sort(key=lambda x: x[1], reverse=True)
    
#     mots = []
    
#     for word in top_words[:25]:
#         mots.append(word[0])
        
#     return (mots)


def polarisation_mots(mots):
    # Normalisation :
    df = pd.read_csv("comparatif_Fire Tablet, 7 Display, Wi-Fi, 8 GB.csv", low_memory=False)
    df.fillna('', inplace=True)
    st.write('check')
    # scaler = MinMaxScaler()
    # columns_to_normalize = df.columns[21:28]
    # df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    
    tab = np.array([['mots', 'pos', 'neu', 'neg']])
    
    for mot in mots:
        count_pos = 0
        count_neg = 0
        count_neu = 0
        for i, com in enumerate(df['text']):
            mot = str(mot)
            if mot in com:
                if df.loc[i, 'rating'] == 5:
                    count_neg += 1
            elif df.loc[i, 'rating'] == 1 or 2 or 3:
                count_pos += 1
            else :
                count_neu += 1
            # st.write(com, "=", count_pos, count_neg, count_neu)
        count = count_neg + count_pos + count_neu
        if count == 0:
            count = 0.00000001
        new = [mot, count_pos/count, count_neu/count, count_neg/count]
        tab = np.vstack((tab, new))
        
    # st.write(tab)
    return tab


# ----------
def foo():
    mots = ['chat', 'chien', 'maison', 'arbre', 'soleil', 'ordinateur', 'jardin', 'musique', 'voiture', 'plage']
    scores = {
        'chat': {'pos': 0.8, 'neutral': 0.1, 'neg': 0.1},
        'chien': {'pos': 0.7, 'neutral': 0.2, 'neg': 0.1},
        'maison': {'pos': 0.6, 'neutral': 0.3, 'neg': 0.1},
        'arbre': {'pos': 0.5, 'neutral': 0.4, 'neg': 0.1},
        'soleil': {'pos': 0.9, 'neutral': 0.05, 'neg': 0.05},
        'ordinateur': {'pos': 0.3, 'neutral': 0.6, 'neg': 0.1},
        'jardin': {'pos': 0.7, 'neutral': 0.2, 'neg': 0.1},
        'musique': {'pos': 0.8, 'neutral': 0.1, 'neg': 0.1},
        'voiture': {'pos': 0.4, 'neutral': 0.4, 'neg': 0.2},
        'plage': {'pos': 0.9, 'neutral': 0.05, 'neg': 0.05}
    }
    style = """
    <style>
        .score-bar {
            height: 100%;
            border-radius: 5px;
            margin-right: 4px;
        }
        .score-bar:last-child {
            margin-right: 0;
        }
        .highlighted-text {
            font-size: 18px;
            font-weight: bold;
        }
        .legend-container {
            display: flex;
            flex-direction: row;
            gap: 50px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            position: relative;
            border-radius: 5px;
            top: -6px;
        }
    </style>
    """

    st.markdown(style, unsafe_allow_html=True)
    

    for mot in mots:
        if mot in scores:
            pos_score = int(scores[mot]['pos'] * 100)
            neutral_score = int(scores[mot]['neutral'] * 100)
            neg_score = int(scores[mot]['neg'] * 100)
            html_string = f"""
            <div style="display: flex; align-items: center;">
                <p class="highlighted-text" style="width: 120px; margin-right: 10px;">{mot.capitalize()}</p>
                <div style="display: flex; width: 60%; height: 20px;">
                    <div class="score-bar" style="background-color: green; width: {pos_score}%;"></div>
                    <div class="score-bar" style="background-color: orange; width: {neutral_score}%;"></div>
                    <div class="score-bar" style="background-color: red; width: {neg_score}%;"></div>
                </div>
            </div>
            """
            st.markdown(html_string, unsafe_allow_html=True)

    html_legend = """
        <div class="legend-container">
            <div class="legend-item">
                <div class="legend-color" style="background-color: green;"></div>
                <p class="legend-text">Positif</p>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: orange;"></div>
                <p class="legend-text">Neutre</p>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: red;"></div>
                <p class="legend-text">Négatif</p>
            </div>
        </div>
    """
    st.write(html_legend, unsafe_allow_html=True)


def foo2(scores):
    scores = scores[1:]
    st.write(scores[0][1])
    
    style = """
    <style>
        .score-bar {
            height: 100%;
            border-radius: 5px;
            margin-right: 4px;
        }
        .score-bar:last-child {
            margin-right: 0;
        }
        .highlighted-text {
            font-size: 18px;
            font-weight: bold;
        }
        .legend-container {
            display: flex;
            flex-direction: row;
            gap: 50px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            position: relative;
            border-radius: 5px;
            top: -6px;
        }
    </style>
    """

    st.markdown(style, unsafe_allow_html=True)

    for score in scores:
        mot = score[0]
        pos_score = int(score[1] * 100)
        neutral_score = int(score[2] * 100)
        neg_score = int(score[3] * 100)
        html_string = f"""
        <div style="display: flex; align-items: center;">
            <p class="highlighted-text" style="width: 120px; margin-right: 10px;">{mot.capitalize()}</p>
            <div style="display: flex; width: 60%; height: 20px;">
                <div class="score-bar" style="background-color: green; width: {pos_score}%;"></div>
                <div class="score-bar" style="background-color: orange; width: {neutral_score}%;"></div>
                <div class="score-bar" style="background-color: red; width: {neg_score}%;"></div>
            </div>
        </div>
        """
        st.markdown(html_string, unsafe_allow_html=True)

    html_legend = """
        <div class="legend-container">
            <div class="legend-item">
                <div class="legend-color" style="background-color: green;"></div>
                <p class="legend-text">Positif</p>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: orange;"></div>
                <p class="legend-text">Neutre</p>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: red;"></div>
                <p class="legend-text">Négatif</p>
            </div>
        </div>
    """
    st.write(html_legend, unsafe_allow_html=True)


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

def generate_wordcloud(mots):
    st.title("Nuage de mots")
    wordcloud = WordCloud(width=800, height=400, background_color='white', relative_scaling=0.5, max_words=50, prefer_horizontal=0.7).generate_from_frequencies(mots)
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
COLOR_MAPPING_SENTIMENTS = {"roberta_neg": "red", "roberta_pos": "green"}

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
    display_reviews(df.nlargest(n, "roberta_pos"), keywords, "roberta_pos", "green")
    display_reviews(df.nlargest(n, "roberta_neg"), keywords, "roberta_neg", "red")

def display_reviews(df, keywords, sentiment, color):
    map = {'roberta_pos': 'positive', 'roberta_neg': 'negative'}
    st.header(f"Top {len(df)} {sentiment} reviews")
    for index, row in df.iterrows():
        title, text, rating = row["title"].capitalize(), row["text"], int(row["rating"])
        write_review_w_keywords(title, text, rating, keywords, COLORS[color])
        st.write(f"This review got a :{color}[{round(row[sentiment], 3)}] {map[sentiment]} score.")

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
    st.write("---")

    df_filtered, keywords = get_df_filtered(df, 3)
    n = st.number_input("Number of helpful reviews to show", min_value=1, max_value=15, value=3)
    st.write("---")

    display_top_helpful_comments(df_filtered, n, keywords)
    st.write("---")
    display_top_pos_neg_reviews(df_filtered, n, keywords)
    st.write("---")
    st.header("Sentiments distribution")
    plot_sentiments_distribution(df, ["roberta_pos", "roberta_neg"], COLOR_MAPPING_SENTIMENTS)
    st.write("---")
    st.header("Sentiments per rating")
    plot_sentiments_per_month_min_max(df, ["roberta_pos", "roberta_neg"], COLOR_MAPPING_SENTIMENTS, 40)
    # display_emotions_sentiments_analysis(df)
    st.write('---')
    mots = bigrames(df)
    generate_wordcloud(mots)

def analysis():
    # --- CSS ---
    page_bg_img = """
        <style>
            [class="main css-uf99v8 e1g8pov65"]{
                background-color: #ffffff;
            }
        </style>
        """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    # --- CODE ---
    st.title("Analysis")
    if "df" in st.session_state:
        df = st.session_state["df"]
        df = df.rename(columns={'pos': 'positive', 'neg': 'negative'})
        analysis_page(df)
    else:
        st.write("No Data uploaded")

