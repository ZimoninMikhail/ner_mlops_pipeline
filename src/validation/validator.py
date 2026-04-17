"""
Модуль для валидации модели на данных.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from transformers import AutoModelForTokenClassification

from .metrics import compute_ner_metrics, extract_labels_from_tensors


class NERValidator:
    """
    Валидатор для NER модели.
    """
    
    def __init__(self, model_path: Path, device: str = "cpu"):
        """
        Args:
            model_path: Путь к сохранённой модели
            device: Устройство ("cpu" или "cuda")
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.id2label = None
        self._load_model()
    
    def _load_model(self):
        """Загружает модель и маппинг меток."""
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.id2label = self.model.config.id2label
        print(f"Модель загружена: {self.model_path}")
        print(f"  num_labels: {len(self.id2label)}")
    
    def validate(
        self,
        dataloader,
        ignore_label: int = -100
    ) -> Dict[str, Any]:
        """
        Валидация модели на данных.
        
        Args:
            dataloader: DataLoader с валидационными данными
            ignore_label: ID игнорируемых токенов
            
        Returns:
            Словарь с метриками
        """
        all_predictions = []
        all_labels = []
        
        print("Запуск валидации...")
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].cpu().numpy()
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                predictions = outputs.logits.cpu().numpy()
                
                true_labels, pred_labels = extract_labels_from_tensors(
                    predictions, labels, self.id2label, ignore_label
                )
                
                all_labels.extend(true_labels)
                all_predictions.extend(pred_labels)
        
        metrics = compute_ner_metrics(all_labels, all_predictions)
        print(f"  F1-score: {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        
        return metrics
    
    def validate_on_batch(
        self,
        prepared_batch_data: Dict[str, torch.Tensor],
        ignore_label: int = -100
    ) -> Dict[str, Any]:
        """
        Валидация на одном подготовленном батче.
        
        Args:
            prepared_batch_data: Данные из prepared_storage
            ignore_label: ID игнорируемых токенов
            
        Returns:
            Словарь с метриками
        """
        tensors = prepared_batch_data['tensors']
        
        input_ids = tensors['input_ids'].to(self.device)
        attention_mask = tensors['attention_mask'].to(self.device)
        labels = tensors['labels'].cpu().numpy()
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.cpu().numpy()
        
        true_labels, pred_labels = extract_labels_from_tensors(
            predictions, labels, self.id2label, ignore_label
        )
        
        return compute_ner_metrics(true_labels, pred_labels)


def validate_model(
    model_path: Path,
    prepared_storage,
    batch_path: Path,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Упрощённая функция валидации модели.
    
    Args:
        model_path: Путь к модели
        prepared_storage: Экземпляр PreparedDataStorage
        batch_path: Путь к prepared батчу для валидации
        device: Устройство
        
    Returns:
        Словарь с метриками
    """
    validator = NERValidator(model_path, device)
    batch_data = prepared_storage.load_batch(batch_path)
    metrics = validator.validate_on_batch(batch_data)
    
    return metrics