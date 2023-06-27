import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

def home():
    st.title('Welcome to Our Application')
    st.write("This is the home page of our application.")
    st.title("Charger les donnees depuis le dossier 'final_data'")
    get_uploaded_file()

# def get_uploaded_file():
#     uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file, parse_dates=['date', 'dateSeen'])
#         st.session_state['name'] = uploaded_file.name[:-4]


#         start_time = time.time()
#         df_with_scores = modelize(df, 'gpu')


#         st.session_state['df'] = df_with_scores

#         end_time = time.time()
#         execution_time = end_time - start_time

#         st.write(f"Execution time: {execution_time} seconds")

import os
import pickle
import pandas as pd
import streamlit as st
import time

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
            st.write('Data loaded from cache')
        else:
            start_time = time.time()
            df_with_scores = modelize(df, 'gpu')

            with open(pickle_file, 'wb') as f:
                pickle.dump(df_with_scores, f)

            end_time = time.time()
            execution_time = end_time - start_time
            st.write(f"Execution time: {execution_time} seconds")

        st.session_state['df'] = df_with_scores
        st.write(df_with_scores.head())

def modelize_raw(df, device_str):
    device = 0 if device_str == 'gpu' else -1
    from tqdm.auto import tqdm
    # VADERS
    st.write('VADERS')
    nltk.download('vader_lexicon')

    sia = SentimentIntensityAnalyzer()
    res = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row['title'] + ' ' + row['text']
        res[i] = sia.polarity_scores(text)
    
    vaders = pd.DataFrame(res).T
    vaders = vaders.add_prefix('vaders_')
    df = df.join(vaders)

    # ROBERTA
    st.write('ROBERTA')
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    from math import ceil
    from transformers import pipeline

    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    num_batch = 100  

    classifier = pipeline(
        "text-classification",
        model=MODEL,
        tokenizer=MODEL,
        max_length=512,
        truncation=True,
        padding=True,
        top_k=None,
        device=device
    )

    texts = (df['title'] + " " + df['text']).tolist()

    batch_size = ceil(len(texts) / num_batch)  

    sentiment_scores = []
    for text_batch in tqdm(batch(texts, batch_size), total=num_batch, desc='Processing batches'):
        sentiment_scores.extend(classifier(text_batch))

    roberta = pd.DataFrame({
        'roberta_neg': [d[2]['score'] for d in sentiment_scores],
        'roberta_neu': [d[1]['score'] for d in sentiment_scores],
        'roberta_pos': [d[0]['score'] for d in sentiment_scores],
    })

    df = df.join(roberta)

    st.write(df.head())


def modelize(df, device_str):
    device = 0 if device_str == 'gpu' else -1
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    import pandas as pd
    from math import ceil
    from transformers import pipeline
    import streamlit as st

    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    # VADERS
    st.write('VADERS')
    nltk.download('vader_lexicon')

    sia = SentimentIntensityAnalyzer()
    res = {}
    progress_bar = st.progress(0)
    for i, row in enumerate(df.iterrows()):
        text = row[1]['title'] + ' ' + row[1]['text']
        res[i] = sia.polarity_scores(text)
        progress_bar.progress((i + 1) / len(df))

    vaders = pd.DataFrame(res).T
    vaders = vaders.add_prefix('vaders_')
    df = df.join(vaders)

    # ROBERTA
    st.write('ROBERTA')
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"

    num_batch = 100  

    classifier = pipeline(
        "text-classification",
        model=MODEL,
        tokenizer=MODEL,
        max_length=512,
        truncation=True,
        padding=True,
        top_k=None,
        device=device
    )

    texts = (df['title'] + " " + df['text']).tolist()

    batch_size = ceil(len(texts) / num_batch)  

    sentiment_scores = []
    progress_bar = st.progress(0)
    for i, text_batch in enumerate(batch(texts, batch_size)):
        sentiment_scores.extend(classifier(text_batch))
        progress_bar.progress((i + 1) / num_batch)

    roberta = pd.DataFrame({
        'roberta_neg': [d[2]['score'] for d in sentiment_scores],
        'roberta_neu': [d[1]['score'] for d in sentiment_scores],
        'roberta_pos': [d[0]['score'] for d in sentiment_scores],
    })

    df = df.join(roberta)
    progress_bar.progress(1.0)

    st.write(df.head())
    st.write('Go to analysis page')
    return df

