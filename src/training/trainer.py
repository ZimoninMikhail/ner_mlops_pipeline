"""
Модуль для обучения BERT модели.
"""

import torch
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from typing import Dict, Any, Optional
import json
from pathlib import Path
import shutil


class NERDataset(Dataset):
    """
    PyTorch Dataset для NER данных из prepared батча.
    """
    
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self.input_ids = tensors['input_ids']
        self.attention_mask = tensors['attention_mask']
        self.labels = tensors['labels']
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }


def train_model(
    model,
    train_dataset: NERDataset,
    eval_dataset: Optional[NERDataset] = None,
    output_dir: str = "./data/models/temp",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    logging_steps: int = 10,
    eval_steps: int = 50,
    save_steps: int = 100
) -> Trainer:
    """
    Обучает модель NER.
    """
    print("\n" + "="*50)
    print("Настройка обучения")
    print("="*50)
    print(f"  epochs: {num_epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  output_dir: {output_dir}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    logging_steps=logging_steps,
    evaluation_strategy="steps" if eval_dataset else "no", 
    eval_steps=eval_steps if eval_dataset else None,
    save_steps=save_steps,
    save_total_limit=2,
    load_best_model_at_end=True if eval_dataset else False,
    report_to="none",
    fp16=False,
)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset else None,
    )
    
    print("\n" + "="*50)
    print("Начало обучения")
    print("="*50)
    
    trainer.train()
    
    print("\n" + "="*50)
    print("Обучение завершено!")
    print("="*50)
    
    return trainer


def train_from_prepared_batch(
    prepared_storage,
    batch_path: Path,
    model_name: str = "DeepPavlov/rubert-base-cased",
    models_dir: str = "./data/models",
    num_epochs: int = 1,
    batch_size: int = 4
) -> tuple:
    """
    Загружает подготовленный батч и обучает модель.
    
    Returns:
        (model_path, version) - путь к сохранённой модели и версия
    """
    from .model_factory import create_model, get_device
    from datetime import datetime
    
    print("\n" + "="*50)
    print("Загрузка подготовленных данных")
    print("="*50)
    
    batch_data = prepared_storage.load_batch(batch_path)
    tensors = batch_data['tensors']
    dataset_info = batch_data['dataset_info']
    
    print(f"  num_samples: {tensors['input_ids'].shape[0]}")
    print(f"  max_length: {tensors['input_ids'].shape[1]}")
    print(f"  num_labels: {dataset_info['num_labels']}")
    
    train_dataset = NERDataset(tensors)
    
    device = get_device()
    model = create_model(
        model_name=model_name,
        num_labels=dataset_info['num_labels'],
        label2id=dataset_info.get('label2id'),
        id2label=dataset_info.get('id2label'),
        device=device
    )
    
    temp_dir = Path("./data/models/.temp_checkpoints")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = train_model(
        model=model,
        train_dataset=train_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        output_dir=str(temp_dir)
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    version = f"v{timestamp}"
    final_model_path = Path(models_dir) / version
    final_model_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(final_model_path)
    
    metadata = {
        'version': version,
        'source_batch': str(batch_path.name),
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'num_samples': len(train_dataset),
        'model_name': model_name,
        'num_labels': dataset_info['num_labels'],
        'timestamp': timestamp
    }
    with open(final_model_path / "training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    print(f"\n✅ Модель сохранена: {final_model_path}")
    print(f"   Версия: {version}")
    
    return str(final_model_path), version