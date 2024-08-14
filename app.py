import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('punkt_tab')


# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

st.title('Spam Checker')
input_sms = st.text_input('ENTER THE MESSAGE')


def text_transform(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text into words
    text = nltk.word_tokenize(text)
    y = []
    # Keep only alphanumeric tokens
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    # Apply stemming
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


# Load pre-trained models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

if st.button('Check'):
    transform_sms = text_transform(input_sms)
    vector_input = tfidf.transform([transform_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
