"""
Модуль обслуживания модели (инференс).
"""

from .model_loader import ModelLoader
from .predictor import NERPredictor
from .inference_pipeline import InferencePipeline

__all__ = [
    'ModelLoader',
    'NERPredictor',
    'InferencePipeline'
]