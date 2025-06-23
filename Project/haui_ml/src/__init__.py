# src package init
# Import text_process_pipeline to make it available globally for pickle compatibility
from src.utils.text_preprocessing import text_process_pipeline

__all__ = ['text_process_pipeline']
