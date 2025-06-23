#Chua cac cau hinh, duong dan can thiet
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Model paths
MODEL_DIR = BASE_DIR / "model_weights"

# Available models configuration
AVAILABLE_MODELS = {
    "Bag of Words Models": {
        "Logistic Regression": {
            "path": MODEL_DIR / "bow_logistic_regression_model.pkl",
            "vectorizer": MODEL_DIR / "bow_vectorizer.pkl",
            "type": "bow",
            "description": "Linear model with good interpretability and fast inference"
        },
        "Random Forest": {
            "path": MODEL_DIR / "bow_random_forest_model.pkl",
            "vectorizer": MODEL_DIR / "bow_vectorizer.pkl",
            "type": "bow",
            "description": "Ensemble method with high accuracy and feature importance"
        },
        "Naive Bayes": {
            "path": MODEL_DIR / "bow_naive_bayes_model.pkl",
            "vectorizer": MODEL_DIR / "bow_vectorizer.pkl",
            "type": "bow",
            "description": "Probabilistic classifier, very fast training and inference"
        },
        "Decision Tree": {
            "path": MODEL_DIR / "bow_decision_tree_model.pkl",
            "vectorizer": MODEL_DIR / "bow_vectorizer.pkl",
            "type": "bow",
            "description": "Tree-based model with high interpretability"
        },
        "K-Nearest Neighbors": {
            "path": MODEL_DIR / "bow_knn_model.pkl",
            "vectorizer": MODEL_DIR / "bow_vectorizer.pkl",
            "type": "bow",
            "description": "Instance-based learning, good for local patterns"
        },
        "Support Vector Machine": {
            "path": MODEL_DIR / "bow_svm_model.pkl",
            "vectorizer": MODEL_DIR / "bow_vectorizer.pkl",
            "type": "bow",
            "description": "SVM classifier with RBF kernel, excellent for complex patterns"
        }
    },
    "Pipeline Models": {
        "TF-IDF + Logistic Regression": {
            "path": MODEL_DIR / "pipeline_logistic_regression_model.pkl",
            "type": "pipeline",
            "description": "Complete pipeline with TF-IDF transformation and Logistic Regression"
        },
        "TF-IDF + Support Vector Machine": {
            "path": MODEL_DIR / "pipeline_svm_model.pkl",
            "type": "pipeline",
            "description": "Complete pipeline with TF-IDF transformation and SVM classifier"
        }
    }
}

# App configuration
APP_CONFIG = {
    "title": "🔍 Fake Reviews Detection System",
    "sidebar_title": "Model Selection & Settings",
    "max_review_length": 5000,
    "default_model_category": "Bag of Words Models",
    "default_model": "Logistic Regression"
}

# Text processing configuration
TEXT_PROCESSING_CONFIG = {
    "max_features": 5000,
    "remove_stopwords": True,
    "apply_stemming": True,
    "min_word_length": 2
}

# UI configuration
UI_CONFIG = {
    "theme": {
        "primary_color": "#FF6B6B",
        "background_color": "#FFFFFF",
        "secondary_background_color": "#F0F2F6",
        "text_color": "#262730"
    },
    "layout": {
        "sidebar_width": 300,
        "main_width": 700
    }
}

# Performance metrics display
METRICS_CONFIG = {
    "display_metrics": ["confidence", "prediction_time", "model_info"],
    "confidence_threshold": 0.7,
    "show_probability_distribution": True
}
