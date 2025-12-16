import pandas as pd
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# --- 1. EXTRACTION (E) ---
def load_data(url="https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv"):
    """
    Simulates fetching a dataset from an external source.
    Demonstrates ability to handle remote data sources.
    """
    print("--- 1. EXTRACTING DATA ---")
    try:
        # Loading a sample dataset of reviews
        df = pd.read_csv(url).head(1000) 
        df = df[['reviewText']] 
        df.columns = ['raw_text']
        print(f"Successfully loaded {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame({'raw_text': []})

# --- 2. TRANSFORMATION (T) ---
def preprocess_text(text):
    """
    Data Normalization: The most critical step for AI training data.
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Lowercasing and noise removal
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Tokenization and Stopword removal
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    
    return " ".join(filtered_tokens)

def analyze_sentiment(df):
    """
    Algorithmic Classification: Assigns sentiment labels.
    """
    print("\n--- 2. TRANSFORMING DATA (Cleaning & Analysis) ---")
    
    # Ensure NLTK resources are available
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    
    df = df.dropna(subset=['raw_text'])
    
    # Apply cleaning
    tqdm.pandas(desc="Cleaning Text")
    df['cleaned_text'] = df['raw_text'].progress_apply(preprocess_text)
    
    # Initialize Sentiment Analyzer
    sia = SentimentIntensityAnalyzer()
    
    def get_label(text):
        score = sia.polarity_scores(text)['compound']
        if score >= 0.05: return 'Positive'
        elif score <= -0.05: return 'Negative'
        else: return 'Neutral'
    
    tqdm.pandas(desc="Classifying")
    df['sentiment_label'] = df['cleaned_text'].progress_apply(get_label)
    
    return df

# --- 3. LOAD (L) ---
def save_data(df, filename="clean_sentiment_data.csv"):
    """
    Final Output: Creating a clean, verifiable dataset.
    """
    print(f"\n--- 3. LOADING DATA to {filename} ---")
    df.to_csv(filename, index=False)
    print("Pipeline Complete. Reviewing results:")
    print(df[['raw_text', 'sentiment_label']].head())

if __name__ == "__main__":
    raw_df = load_data()
    if not raw_df.empty:
        processed_df = analyze_sentiment(raw_df)
        save_data(processed_df)
