
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the pre-trained model and TF-IDF vectorizer
with open('model1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer1.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

ps = PorterStemmer()

# Function for text transformation
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]

    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    y = [ps.stem(i) for i in text]

    return " ".join(y)

# Streamlit UI
st.title("Email classifier")

input_mail = st.text_input("Enter the mail")

if st.button('Predict'):
    # 1. Preprocessing
    transformed_mail = transform_text(input_mail)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_mail])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not spam")
