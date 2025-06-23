import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    def __init__(self, remove_stopwords=True, apply_stemming=True, min_word_length=2):
        self.remove_stopwords = remove_stopwords
        self.apply_stemming = apply_stemming
        self.min_word_length = min_word_length
        
        # Initialize stemmer
        if self.apply_stemming:
            self.stemmer = PorterStemmer()
        
        # Initialize stopwords (keep 'not' as it's important for sentiment)
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
            if 'not' in self.stop_words:
                self.stop_words.remove('not')
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase and remove punctuation
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        
        # Split into words
        words = text.split()

        # Filter words by length
        words = [word for word in words if len(word) >= self.min_word_length]
        
        # Remove stopwords
        if self.remove_stopwords:
            words = [word for word in words if word not in self.stop_words]

        # happiness -> happi or processor, processing -> process => Mất ngữ cảnh
        # Lemmatization (Nhưng chậm hơn)
        if self.apply_stemming:
            words = [self.stemmer.stem(word) for word in words]
        
        return ' '.join(words)
    
    def preprocess_for_bow(self, text):
        return self.clean_text(text)
    
    def preprocess_for_pipeline(self, text):
        if not isinstance(text, str):
            return ""

        return text.strip()

def validate_text_input(text, max_length=5000):
    if not text or not text.strip():
        return False, "Please enter some text to analyze."
    
    if len(text) > max_length:
        return False, f"Text is too long. Maximum length is {max_length} characters."
    
    if len(text.split()) < 3:
        return False, "Please enter at least 3 words for meaningful analysis."
    
    return True, ""

#Trích xuất đặc trưng thống kê từ văn bản: Số ký tự, số từ, số câu, do dai TB,
# Tỷ lệ chữ in hoa/ thường, tỷ lệ dấu câu
def extract_text_features(text):
    if not isinstance(text, str):
        return {}
    
    words = text.split()
    sentences = text.split('.')
    
    features = {
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,

        #Giúp nhận diện văn phong, cxuc, spam. VD: Sử dụng nhieu !!! thì rất dễ là spam
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'punctuation_ratio': sum(1 for c in text if c in string.punctuation) / len(text) if text else 0
    }
    
    return features

# Global function for pipeline compatibility
def text_process_pipeline(text):
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_for_pipeline(text)

