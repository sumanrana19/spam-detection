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
st.set_page_config(page_title="Email Classifier", page_icon="📧")
st.title("Email Classifier")

input_mail = st.text_input("Enter the email")

if st.button('Predict'):
    # 1. Preprocessing
    transformed_mail = transform_text(input_mail)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_mail])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.error("This email is likely Spam 🚨")
    else:
        st.success("This email is Not Spam 🎉")

# Sidebar with information
st.sidebar.title("About")
st.sidebar.info(
    "This is a simple email classifier that predicts whether an email is spam or not spam. "
    "It uses a machine learning model trained on a dataset of emails."
)
