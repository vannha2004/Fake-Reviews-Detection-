
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pickle
import time
import logging
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, Any, Optional

from src.config.config import AVAILABLE_MODELS
from src.utils.text_preprocessing import text_process_pipeline, TextPreprocessor

import sys
import builtins

sys.modules[__name__].text_process_pipeline = text_process_pipeline
builtins.text_process_pipeline = text_process_pipeline

# Also make it available in __main__ module if running as script
if hasattr(sys.modules.get('__main__'), '__file__'):
    sys.modules['__main__'].text_process_pipeline = text_process_pipeline

# Sử dụng log thay vì print vì log lưu ra file -> dễ bảo trì hơn
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) #Ghi lại ttin về qtrinh tải/ dự đoán

#Lớp chính để tải, dự đoán, quản lý mô hình
class ModelManager:
    
    def __init__(self):
        self.loaded_models = {} #Lưu mô hình đã nạp
        self.loaded_vectorizers = {} #Lưu vector t.ứ với mô hình kiểu Bow/ TF-IDF
        self.text_preprocessor = TextPreprocessor() #XL văn bản đầu vào


    def load_model(self, model_category: str, model_name: str) -> bool:
        try:
            model_key = f"{model_category}_{model_name}"
            
            if model_key in self.loaded_models:
                logger.info(f"Model {model_key} already loaded")
                return True

            #AVAILABLE_MODELS: Các module đã có sẵn trong config.py
            model_config = AVAILABLE_MODELS[model_category][model_name]
            model_path = model_config["path"]
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load the main model
            # Ensure text_process_pipeline is available globally for pipeline models
            if model_config['type'] == 'pipeline':
                # Make function available in multiple namespaces for pickle compatibility
                import builtins
                import __main__
                
                builtins.text_process_pipeline = text_process_pipeline
                sys.modules['__main__'].text_process_pipeline = text_process_pipeline
                
                # Also try to set it in the main module's globals
                if hasattr(__main__, '__dict__'):
                    __main__.__dict__['text_process_pipeline'] = text_process_pipeline
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f) #Load các mô hình đã train
            
            self.loaded_models[model_key] = {
                'model': model,
                'config': model_config,
                'type': model_config['type']
            }
            
            # Load vectorizer for BoW models
            if model_config['type'] == 'bow':
                vectorizer_path = model_config['vectorizer']
                if vectorizer_path.exists():
                    with open(vectorizer_path, 'rb') as f:
                        vectorizer = pickle.load(f) #Load các mô hình đã train
                    self.loaded_vectorizers[model_key] = vectorizer
                    logger.info(f"Vectorizer loaded for {model_key}")
                else:
                    logger.error(f"Vectorizer not found: {vectorizer_path}")
                    return False
            
            logger.info(f"Successfully loaded model: {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_category}_{model_name}: {str(e)}")
            return False

    def predict(self, text: str, model_category: str, model_name: str) -> Dict[str, Any]:
        model_key = f"{model_category}_{model_name}"
        
        # Load model if not already loaded
        if model_key not in self.loaded_models:
            if not self.load_model(model_category, model_name):
                return self._create_error_result("Failed to load model")
        
        try:
            start_time = time.time()
            model_info = self.loaded_models[model_key]
            model = model_info['model']
            model_type = model_info['type']
            
            # Tiền XL VB
            if model_type == 'bow':
                processed_text = self.text_preprocessor.preprocess_for_bow(text)
                vectorizer = self.loaded_vectorizers[model_key]

                text_vector = vectorizer.transform([processed_text]).toarray()
                
            elif model_type == 'pipeline':
                processed_text = self.text_preprocessor.preprocess_for_pipeline(text)
                text_vector = [processed_text]
            
            else:
                return self._create_error_result(f"Unknown model type: {model_type}")
            
            # Make prediction
            prediction = model.predict(text_vector)[0]
            
            # Get prediction probabilities if available
            try:
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(text_vector)[0]
                    confidence = float(np.max(probabilities))
                    
                    # Get class labels
                    if hasattr(model, 'classes_'):
                        classes = model.classes_
                        prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
                    else:
                        prob_dict = {'OR (Real)': float(probabilities[0]), 'CG (Fake)': float(probabilities[1])}
                        
                elif hasattr(model, 'decision_function'):
                    # For SVM
                    decision = model.decision_function(text_vector)[0]
                    confidence = abs(float(decision))
                    prob_dict = {'confidence_score': confidence}
                else:
                    confidence = 0.5
                    prob_dict = {}
                    
            except Exception as e:
                logger.warning(f"Could not get probabilities: {str(e)}")
                confidence = 0.5
                prob_dict = {}
            
            prediction_time = time.time() - start_time
            
            # Format result
            result = {
                'prediction': str(prediction),
                'prediction_label': 'Real Review' if str(prediction) == 'OR' else 'Fake Review',
                'confidence': confidence,
                'probabilities': prob_dict,
                'prediction_time': prediction_time,
                'model_info': {
                    'name': model_name,
                    'category': model_category,
                    'type': model_type,
                    'description': model_info['config']['description']
                },
                'processed_text': processed_text[:100] + "..." if len(processed_text) > 100 else processed_text,
                'success': True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return self._create_error_result(f"Prediction failed: {str(e)}")
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        return {
            'prediction': 'Error',
            'prediction_label': 'Error',
            'confidence': 0.0,
            'probabilities': {},
            'prediction_time': 0.0,
            'model_info': {},
            'processed_text': '',
            'success': False,
            'error': error_message
        }
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models"""
        return AVAILABLE_MODELS
    
    def get_model_info(self, model_category: str, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        try:
            return AVAILABLE_MODELS[model_category][model_name]
        except KeyError:
            return {}
    
    def unload_model(self, model_category: str, model_name: str) -> bool:
        """Unload a specific model from memory"""
        model_key = f"{model_category}_{model_name}"
        
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            
        if model_key in self.loaded_vectorizers:
            del self.loaded_vectorizers[model_key]
            
        logger.info(f"Unloaded model: {model_key}")
        return True
    
    def get_loaded_models(self) -> list:
        """Get list of currently loaded models"""
        return list(self.loaded_models.keys())
