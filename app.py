import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(Text):

    Text = Text.lower()            # Making text lowercase
    Text = nltk.word_tokenize(Text) #

    y = []
    for i in Text:
        if i.isalnum():

            y.append(i)

    Text = y[:]
    y.clear()


    for i in Text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    Text = y[:]
    y.clear()

    for i in Text:
        y.append(ps.stem(i))


    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. Preprocess
    transform_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transform_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Beware, It is 'SPAM!' ")
    else:
        st.header("It is 'SAFE!' ")
