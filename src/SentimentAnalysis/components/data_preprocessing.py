import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from src.SentimentAnalysis.logging.logger import logging
from src.SentimentAnalysis.utils.common import save_pickle

nltk.download('stopwords')

class DataPreprocessing:
    def __init__(self, config):
        self.config = config
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()  # Initialize stemmer

    def clean_text(self, text):
        """
        Clean and preprocess text:
        1. Remove punctuation.
        2. Convert to lowercase.
        3. Remove stopwords.
        4. Apply stemming.
        """
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()  # Convert to lowercase
        text = ' '.join([self.stemmer.stem(word) for word in text.split() if word not in self.stop_words])  # Stemming
        return text

    def vectorize_text(self, train_df, test_df):
        """
        Vectorize text data using TF-IDF.
        """
        # Clean text data
        train_df['review'] = train_df['review'].apply(self.clean_text)
        test_df['review'] = test_df['review'].apply(self.clean_text)

        # Vectorize text data
        vectorizer = TfidfVectorizer(max_features=self.config.max_features)
        X_train = vectorizer.fit_transform(train_df['review'])
        X_test = vectorizer.transform(test_df['review'])

        # Save the vectorizer
        save_pickle(vectorizer, self.config.vectorizer_path)
        logging.info("Text vectorization completed.")
        return X_train, X_test