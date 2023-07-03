import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import plotly.graph_objects as go

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
import os
import pickle
import pandas as pd
import streamlit as st
import time
from sklearn.feature_extraction.text import TfidfVectorizer




def home():
    st.title('Welcome to Our Application')
    st.write("This is the home page of our application.")
    st.title("Charger les donnees depuis le dossier 'example_of_products'")
    get_uploaded_file()

def get_uploaded_file():
    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['date', 'dateSeen'])
        name =  uploaded_file.name[:-4]

        st.session_state['name'] = name
        os.makedirs('cache', exist_ok=True)  
        pickle_file = f'cache/{name}.pickle'
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as f:
                df_with_scores = pickle.load(f)
            st.write('Loading from cache..')
            st.write('Pro Analyzer Model')
            progress_bar = st.progress(0)
            progress_bar.progress(100)

            st.write('Word cloud')
            progress_bar = st.progress(0)
            progress_bar.progress(100)
            st.write('Vader Lexicon Model')
            progress_bar = st.progress(0)
            progress_bar.progress(100)
        else:
            start_time = time.time()
            df_with_scores = modelize(df)

            with open(pickle_file, 'wb') as f:
                pickle.dump(df_with_scores, f)

            end_time = time.time()
            execution_time = end_time - start_time
            st.write(f"Execution time: {execution_time:.1f} seconds")

        st.session_state['df'] = df_with_scores





def modelize(df):
    df = df.dropna().reset_index()
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    import pandas as pd
    from math import ceil
    from transformers import pipeline
    import streamlit as st

    # VADERS
    st.write('Vader Lexicon Model')
    nltk.download('vader_lexicon')

    sia = SentimentIntensityAnalyzer()
    res = {}
    progress_bar = st.progress(0)
    for i, row in enumerate(df.iterrows()):
        text = str(row[1]['title']) + ' ' + str(row[1]['text'])
        res[i] = sia.polarity_scores(text)
        progress_bar.progress((i + 1) / len(df))

    vaders = pd.DataFrame(res).T
    vaders = vaders.add_prefix('vaders_')
    df = df.join(vaders)

    # ROBERTA
    st.write('Pro Analyzer Model')

    import torch
    import pandas as pd
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the saved model and move it to the GPU
    model = DistilBertForSequenceClassification.from_pretrained('./saved_model')
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Create a tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Texts to analyze
    texts = (df['title'] + " " + df['text']).tolist()

    # Batch processing
    batch_size = 8
    total_texts = len(texts)
    start_index = 0
    positive_scores, neutral_scores, negative_scores = [], [], []

    # Adding the progress bar
    progress_bar = st.progress(0)

    while start_index < total_texts:
        end_index = min(start_index + batch_size, total_texts)
        batch_texts = texts[start_index:end_index]

        # Tokenization of batch texts
        encodings = tokenizer(batch_texts, truncation=True, padding=True)

        # Preparation of input tensors and move them to the GPU
        input_ids = torch.tensor(encodings['input_ids']).to(device)
        attention_mask = torch.tensor(encodings['attention_mask']).to(device)

        # Make predictions
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # Move the predicted probabilities to the CPU
        predicted_probabilities = torch.softmax(outputs.logits, dim=1).cpu()

        # Extract positive, neutral, and negative scores
        positive_scores.extend(predicted_probabilities[:, 0].tolist())
        neutral_scores.extend(predicted_probabilities[:, 1].tolist())
        negative_scores.extend(predicted_probabilities[:, 2].tolist())

        # Clear intermediate variables
        del input_ids, attention_mask, encodings, predicted_probabilities
        torch.cuda.empty_cache()

        start_index = end_index

        # Update the progress bar
        progress_bar.progress(start_index / total_texts)

    # Add the results to the DataFrame
    df['roberta_pos'] = positive_scores
    df['roberta_neu'] = neutral_scores
    df['roberta_neg'] = negative_scores
    


    st.write('Word cloud')
    progress_bar = st.progress(0)
    progress_bar.progress(20)
    mots = bigrames(df)
    progress_bar.progress(50)
    generate_wordcloud(mots)
    progress_bar.progress(100)

    st.write('Go to analysis page')
    
    return df


def bigrames(df):
    data = df['text']
    data = data.fillna('')
    nltk.download('stopwords')
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
    wordcloud = WordCloud(width=800, height=400, background_color='white', relative_scaling=0.5, max_words=50, prefer_horizontal=0.7).generate_from_frequencies(mots)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    name = st.session_state['name']
    plt.savefig(f'word_clouds/{name}.png')