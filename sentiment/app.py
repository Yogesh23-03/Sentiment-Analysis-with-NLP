import streamlit as st
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')

# Load the trained model and vectorizer
load = pickle.load(open('trained_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

pstemer = PorterStemmer()

# Function for stemming
def stemming(content):
    stcontent = re.sub('[^a-zA-Z]', ' ', content)
    stcontent = stcontent.lower()
    stcontent = stcontent.split()
    stcontent = [pstemer.stem(word) for word in stcontent if not word in stopwords.words('english')]
    stcontent = ' '.join(stcontent)
    return stcontent

# Streamlit app
st.title("Sentiment Analysis App")

# Text input for user
input_text = st.text_area("Enter text for sentiment analysis:")

# Analyze button
if st.button("Analyze"):
    # Preprocess the input text
    processed_text = stemming(input_text)

    # Vectorize the text
    vectorized_text = vectorizer.transform([processed_text])

    # Make prediction
    prediction = load.predict(vectorized_text)

    # Display the prediction
    if prediction == 1:
        st.write("Sentiment: Positive")
    else:
        st.write("Sentiment: Negative")