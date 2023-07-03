import re
import os
import sys
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
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.plots import (
    plot_sentiments_distribution, 
    plot_sentiments_per_month_min_max, 
    generate_sentiments_per_month
)

def monogramme(df):
    nlp = spacy.load('en_core_web_sm') 
    data = df[df['roberta_neg'] >= 0.17]['text']
    data.fillna('', inplace=True)
    
    filtered_texts = [" ".join([token.text for token in nlp(text) if token.pos_ in ["NOUN", "PROPN"]]) for text in data]
    filtered_data = pd.Series(filtered_texts)

    stop_words = stopwords.words('english')
    stop_words.extend(["product",'months', 'year', "amazon", "fire", "kindle", "echo", "alexa","however","get", "great","easy", "loves", "device", "best", "nice", "also", "everything", "would", "really", "much","excellent","one", "two",'awesome',"first", "friday", "good", "well","perfect","something","time","even","could","like","lot","happy","things","still","anyone","another","highly","definitely","day"])

    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 1))
    X = vectorizer.fit_transform(filtered_data)
    feature_names = vectorizer.get_feature_names_out()


    top_words = []
    for idx, word in enumerate(feature_names):
        score = X.getcol(idx).sum()
        top_words.append((word, score))
    top_words.sort(key=lambda x: x[1], reverse=True)
    
    mots = []
    
    for word in top_words[:25]:
        mots.append(word[0])
        
    return (mots)


def polarisation_mots(df , mots, polarity):
    df.fillna('', inplace=True)
    
    tab = [['mots', 'pos', 'neu', 'neg']]
    
    for mot in mots:
        count_pos = 0
        count_neg = 0
        count_neu = 0
        for i, com in enumerate(df['text']):
            mot = str(mot)
            if mot in com:
                if df.loc[i, 'vaders_compound'] > 0.7:
                    count_pos += 1
                elif df.loc[i, 'vaders_compound'] < 0.3:
                    count_neg += 1
                else :
                    count_neu += 1
        count = count_neg + count_pos + count_neu
        new = [mot, count_pos/count, count_neu/count, count_neg/count*3]
        tab.append(new)

    if polarity == 'negative':       
        tab_sort = sorted(tab[1:], key=lambda x: x[3], reverse=True)
    else:
        tab_sort = sorted(tab[1:], key=lambda x: x[1], reverse=True)
    return(tab_sort)

def generate_words_scores(scores):
    scores = scores[1:]
    
    style = """
    <style>
    .score-bar {
        height: 100%;
        border-radius: 5px;
        margin-right: 4px;
        transition: width 0.10s ease-in-out;
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

    # style = """
    # <style>
    #     .score-bar {
    #         height: 100%;
    #         border-radius: 5px;
    #         margin-right: 4px;
    #     }
    #     .score-bar:last-child {
    #         margin-right: 0;
    #     }
    #     .highlighted-text {
    #         font-size: 18px;
    #         font-weight: bold;
    #     }
    #     .legend-container {
    #         display: flex;
    #         flex-direction: row;
    #         gap: 50px;
    #     }
    #     .legend-item {
    #         display: flex;
    #         align-items: center;
    #         gap: 6px;
    #     }
    #     .legend-color {
    #         width: 20px;
    #         height: 20px;
    #         position: relative;
    #         border-radius: 5px;
    #         top: -6px;
    #     }
    # </style>
    # """

    st.markdown(style, unsafe_allow_html=True)

    for score in scores:
        mot = score[0]
        pos_score = int(float(score[1]) * 100)
        neutral_score = int(float(score[2]) * 100)
        neg_score = int(float(score[3]) * 100)
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

def generate_words_scores_gradient_pos(scores):
    scores = scores[1:]

    style = """
    <style>
        .score-bar {
            height: 100%;
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
        .percentage {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-left: 10px;
        }
    </style>
    """

    st.markdown(style, unsafe_allow_html=True)

    for score in scores:
        mot = score[0]
        pos_score = int(float(score[1]) * 100)
        neutral_score = int(float(score[2]) * 100)
        neg_score = int(float(score[3]) * 100)

        total = score[1] + score[2] + score[3]
        normalized_pos = (score[1] / total) * 100
        normalized_neu = (score[2] / total) * 100
        normalized_neg = (score[3] / total) * 100
        gap = 200

        html_string = f"""
        <div style="display: flex; align-items: center;">
            <p class="highlighted-text" style="width: 120px; margin-right: 10px;">{mot.capitalize()}</p>
            <div style="display: flex; width: 60%; height: 20px;">
                <div class="score-bar" style="background: green; width: {pos_score}%;"></div>
                <div class="score-bar" style="background: linear-gradient(to left, yellow, green); width: {gap}px;"></div>  
                <div class="score-bar" style="background: linear-gradient(to left, orange, yellow); width: {gap}px;"></div>  
                <div class="score-bar" style="background: orange; width: {neutral_score}%;"></div>
                <div class="score-bar" style="background: linear-gradient(to left, red, orange); width: {gap}px;"></div>  
                <div class="score-bar" style="background: red; width: {neg_score}%;"></div>
            </div>
            <div class="percentage">
                <div class="legend-item">
                    <div class="legend-color" style="background-color: green;"></div>
                    <span>{normalized_pos:.1f}%</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: orange;"></div>
                    <span>{normalized_neu:.1f}%</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: red;"></div>
                    <span>{normalized_neg:.1f}%</span>
                </div>
            </div>
        </div>
        """
       
        st.markdown(html_string, unsafe_allow_html=True)

def generate_words_scores_gradient_neg(scores):
    scores = scores[1:]

    style = """
    <style>
        .score-bar {
            height: 100%;
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
        .percentage {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-left: 10px;
        }
    </style>
    """

    st.markdown(style, unsafe_allow_html=True)

    for score in scores:
        mot = score[0]
        pos_score = int(float(score[1]) * 100)
        neutral_score = int(float(score[2]) * 100)
        neg_score = int(float(score[3]) * 100)

        total = score[1] + score[2] + score[3]
        normalized_pos = (score[1] / total) * 100
        normalized_neu = (score[2] / total) * 100
        normalized_neg = (score[3] / total) * 100
        gap = 200

        html_string = f"""
        <div style="display: flex; align-items: center;">
            <p class="highlighted-text" style="width: 120px; margin-right: 10px;">{mot.capitalize()}</p>
            <div style="display: flex; width: 60%; height: 20px;">
                <div class="score-bar" style="background: red; width: {neg_score}%;"></div>
                <div class="score-bar" style="background: linear-gradient(to right, red, orange); width: {gap}px;"></div>  
                <div class="score-bar" style="background: orange; width: {neutral_score}%;"></div>
                <div class="score-bar" style="background: linear-gradient(to right, orange, yellow); width: {gap}px;"></div>  
                <div class="score-bar" style="background: linear-gradient(to right, yellow, green); width: {gap}px;"></div>  
                <div class="score-bar" style="background: green; width: {pos_score}%;"></div>
            </div>
            <div class="percentage">
                <div class="legend-item">
                    <div class="legend-color" style="background-color: red;"></div>
                    <span>{normalized_neg:.1f}%</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: orange;"></div>
                    <span>{normalized_neu:.1f}%</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: green;"></div>
                    <span>{normalized_pos:.1f}%</span>
                </div>
            </div>
        </div>
        """
       
        st.markdown(html_string, unsafe_allow_html=True)


def get_top_words(scores, n):
    sorted_scores = sorted(scores, key=lambda x: x[3], reverse=True)
    top_words = [score[0] for score in sorted_scores[:n]]
    return top_words


# --------------------------------------------------------------------------------------------------

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
    st.title("Words clouds")
    st.write('This word cloud provides an overview of how customers feel overall about the product')
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

def get_df_filtered(df, key, words):
    if 'keywords_' not in st.session_state:
        st.session_state.keywords_ = []

    cols = st.columns(len(words))

    for i, word in enumerate(words):
        if cols[i].button(f'Add "{word}"'):
            st.session_state.keywords_.append(word)
    
    keywords_ = st_tags(
        label="Enter Keywords to filter reviews:",
        text="Press enter to add more",
        value=st.session_state.keywords_,
        suggestions=["Price", "Product", "Sales"],
        maxtags=4,
    )
    keywords = [word.lower() for word in keywords_]
    st.session_state.keywords_ = keywords_   # update session state
    df = get_comments_with_keywords(df, keywords)

    st.write(f"{len(df)} comments found containing all the keywords." if not(df.empty) else "There are no comments containing all the keywords.")
    return df, keywords


def write_review(title, text, rating=1, text_font_size=15):
    stars = "★" * rating + "☆" * (5 - rating)
    color = '#FFFFFF'

    # <div style="background: linear-gradient(to right, #262730, #1f1c28); 
    review_html = f"""
    <div style="background: linear-gradient(to right, #a480f3, #7451bd);
                border-radius: 7px; 
                padding: 2px 15px 5px 15px; 
                border: 1px solid #555555; 
                box-shadow: 0 4px 6px 0 hsla(0, 0%, 0%, 0.2);">
        <div style="font-size: 25px; color: {color};">
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
    st.write('')
    st.subheader(f"Top {n} helpful reviews according to the users")
    for index, row in df.nlargest(n, "numHelpful").iterrows():
        title, text, rating = row["title"].capitalize(), row["text"], int(row["rating"])
        write_review_w_keywords(title, text, rating, keywords, COLORS["blue"])
        st.write(f":green[{int(row['numHelpful'])}] people found this review helpful")

def display_top_pos_neg_reviews(df, n, keywords):
    display_reviews(df.nlargest(n, "roberta_pos"), keywords, "roberta_pos", "green")
    display_reviews(df.nlargest(n, "roberta_neg"), keywords, "roberta_neg", "red")

def display_reviews(df, keywords, sentiment, color):
    map = {'roberta_pos': 'positive', 'roberta_neg': 'negative'}
    st.write('')
    st.subheader(f"Top {len(df)} {map[sentiment]} reviews")
    for index, row in df.iterrows():
        title, text, rating = row["title"].capitalize(), row["text"], int(row["rating"])
        write_review_w_keywords(title, text, rating, keywords, COLORS[color])
        st.write(f"This review got a :{color}[{round(row[sentiment], 3)}] {map[sentiment]} score.")

def analysis_page(df):
    name = st.session_state["name"]
    st.subheader(name)
    st.subheader(f"{df.shape[0]} reviews to analyse")
    st.write('---')
    st.title("Words Cloud")
    st.write('This word cloud provides an overview of how customers feel overall about the product')
    image_path = os.path.join("word_clouds", name + ".png")
    st.image(image_path)
    
    st.write("---")
    mots = monogramme(df)
    st.subheader('Top positive words')
    scores_pos = polarisation_mots(df, mots, 'positive')
    generate_words_scores_gradient_pos(scores_pos[:10])

    st.subheader('Top negative words')
    scores_neg = polarisation_mots(df, mots, 'negative')
    generate_words_scores_gradient_neg(scores_neg[:10])

    best_words = get_top_words(scores_neg, 10)

    st.write('---')

    st.title('Custom Reviews Search')
    df_filtered, keywords = get_df_filtered(df, 3, best_words)
    n = st.number_input("Number of reviews to show for each category", min_value=1, max_value=15, value=3)

    display_top_helpful_comments(df_filtered, n, keywords)
    display_top_pos_neg_reviews(df_filtered, n, keywords)


    st.write("---")
    st.header("Sentiments distribution")
    plot_sentiments_distribution(df, ["roberta_pos", "roberta_neg"], COLOR_MAPPING_SENTIMENTS)
    st.write("---")
    st.header("Sentiments per rating")
    plot_sentiments_per_month_min_max(df, ["roberta_pos", "roberta_neg"], COLOR_MAPPING_SENTIMENTS, 40)
    st.write('---')
    
def analysis():
    page_bg_img = """
        <style>
            [class="main css-uf99v8 e1g8pov65"]{
                background-color: #ffffff;
            }
        </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.title("Analysis")
    if "df" in st.session_state:
        df = st.session_state["df"]
        df = df.rename(columns={'pos': 'positive', 'neg': 'negative'})
        analysis_page(df)
    else:
        st.write("No Data uploaded")

