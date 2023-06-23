import streamlit as st
from transformers import pipeline
import torch

def model(df):
    st.write(torch.cuda.is_available())
    model_name = 'j-hartmann/emotion-english-distilroberta-base'
    # cardiffnlp/twitter-roberta-base-sentiment

    classifier = pipeline(
        "text-classification",
        model=model_name,
        truncation=True,
        top_k=None,
        device=0
    )
    sentiment_scores = classifier((df['title']+ " " + df['text']).tolist())
    st.write(sentiment_scores[:3])

print('start')
df = st.session_state['df']
model(df)
print('stop')


