from flask import Flask, render_template, request
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from langdetect import detect, LangDetectException
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Download stopwords for NLTK if not already downloaded
nltk.download('stopwords', quiet=True)

# Initialize stop words and stemmers for both languages
nltk_stop_words = set(stopwords.words("english"))
custom_stop_words = ["pertanyaan", "tentang", "bin", "bte", "mengenai", "enquiry", "berkenaan"]  # Custom Malay stopwords
all_stop_words = nltk_stop_words.union(custom_stop_words)

# Initialize English and Malay stemmers
english_stemmer = PorterStemmer()
factory = StemmerFactory()
malay_stemmer = factory.create_stemmer()

# Preprocess function for multilingual data
def pre_process(df):
    cleaned_data = []
    
    for text in df['Description']:
        try:
            lang = detect(text)  # Detect language (either 'en' or 'id')
        except LangDetectException:
            lang = 'unknown'
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize and remove stop words
        words = text.lower().split()
        
        if lang == 'en':
            filtered_words = [word for word in words if word not in all_stop_words]
            stemmed_words = [english_stemmer.stem(word) for word in filtered_words]  # Stem the words
        elif lang == 'id':  # 'id' stands for Indonesian/Malay language in langdetect
            filtered_words = [word for word in words if word not in custom_stop_words]
            stemmed_words = [malay_stemmer.stem(word) for word in filtered_words]  # Stem the words
        else:
            stemmed_words = words  # If language not recognized, leave the words as is
        
        cleaned_text = " ".join(stemmed_words)
        cleaned_data.append(cleaned_text)

    df['Description'] = cleaned_data
    return df


# Function to generate FAQ
def generate_faq(data, num_clusters=5):
    # Pre-process the inquiries
    data = pre_process(data)

    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(data['Description'])

    # K-Means clustering with random state for reproducibility
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(X)

    # Extract top 5 sentences and words for each cluster
    faq = []
    for cluster_id in range(num_clusters):
        cluster_data = data[data['cluster'] == cluster_id]
        
        # Get top 5 frequent sentences
        top_sentences = cluster_data['Description'].value_counts().nlargest(5).index.tolist()
        
        # Get top 5 frequent words
        words = ' '.join(cluster_data['Description']).split()
        top_words = [word for word, _ in Counter(words).most_common(5)]
        
        faq.append((f"Cluster {cluster_id}", top_sentences, top_words))
    
    return faq

# Internal dataset
def load_internal_data():
    # Read the file internally
    data = pd.read_csv('./artifacts/df.csv')
    return data

@app.route('/', methods=['GET', 'POST'])
def index():
    # Load internal dataset
    data = load_internal_data()

    if request.method == 'POST':
        # Optionally, allow users to specify the number of clusters
        num_clusters = int(request.form.get('num_clusters', 5))  # Default to 5 if not provided
    else:
        num_clusters = 5  # Default value for GET requests
    
    # Generate FAQ with KMeans clustering
    faq = generate_faq(data, num_clusters)
    
    return render_template('faq.html', faq=faq)

if __name__ == '__main__':
    app.run(debug=True)
