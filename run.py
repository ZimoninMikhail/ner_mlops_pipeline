"""
Точка входа в MLOps пайплайн.

Поддерживаемые режимы:
    - update: Полный цикл обработки данных и дообучение модели
    - inference: Применение модели к новым данным
    - summary: Генерация отчёта о работе системы
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np 

sys.path.insert(0, str(Path(__file__).parent))

from src.data_collection.loader import StreamingDataLoader
from src.data_collection.storage import RawDataStorage
from src.data_collection.metadata_calculator import calculate_batch_metadata

from src.data_analysis.parser import parse_batch_entities
from src.data_analysis.type_filter import filter_batch_geo, get_filter_statistics
from src.data_analysis.quality_checker import check_document_quality, calculate_batch_quality
from src.data_analysis.quality_filter import filter_batch_by_quality, get_quality_filter_statistics
from src.data_analysis.processed_data_storage import ProcessedDataStorage

from src.data_preparation.tokenizer_setup import load_tokenizer
from src.data_preparation.bio_encoder import create_label_mapping, encode_batch
from src.data_preparation.prepared_data_storage import PreparedDataStorage

from src.training.trainer import train_from_prepared_batch
from src.validation.model_registry import ModelRegistry
from src.validation.validator import validate_model

from src.serving import InferencePipeline


GEO_TYPES = ["COUNTRY", "CITY", "LOCATION", "STATE_OR_PROV", "DISTRICT"]
MAX_LENGTH = 512
MODEL_NAME = "DeepPavlov/rubert-base-cased"
MIN_CONSISTENCY = 0.95
MIN_ENTITIES = 1

def _make_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    return obj

def run_update():
    """
    Полный пайплайн: сбор → анализ → подготовка → обучение → валидация.
    """
    print("\n" + "="*60)
    print("ЗАПУСК UPDATE ПАЙПЛАЙНА")
    print("="*60)
    
    raw_storage = RawDataStorage("data/raw")
    processed_storage = ProcessedDataStorage("data/processed")
    prepared_storage = PreparedDataStorage("data/prepared")
    
    raw_storage.clear()
    processed_storage.clear()
    prepared_storage.clear()
    
    print("\n1. Загрузка данных...")
    loader = StreamingDataLoader(batch_size=32, shuffle=False)
    loader.load_train()
    print(f"   Всего документов: {len(loader)}")
    
    all_encoded_batches = []
    
    for batch_num, batch in enumerate(loader):
        print(f"\n2. Обработка батча {batch_num + 1}...")
        
        raw_metadata = calculate_batch_metadata(batch)
        raw_path = raw_storage.save_batch(batch, raw_metadata)
        
        parsed_batch = parse_batch_entities(batch)
        geo_batch = filter_batch_geo(parsed_batch, remove_empty_docs=True)
        geo_stats = get_filter_statistics(geo_batch)
        
        quality_batch = [check_document_quality(doc) for doc in geo_batch]
        quality_metrics = calculate_batch_quality(quality_batch)
        
        final_batch = filter_batch_by_quality(
            quality_batch,
            min_consistency=MIN_CONSISTENCY,
            min_entities=MIN_ENTITIES
        )
        quality_filter_stats = get_quality_filter_statistics(final_batch)
        
        processed_metadata = {
            "source_batch": raw_path.name,
            **geo_stats,
            **quality_metrics,
            **quality_filter_stats
        }
        processed_storage.save_batch(final_batch, processed_metadata)
        
        if final_batch:
            tokenizer = load_tokenizer(MODEL_NAME, MAX_LENGTH)
            label2id, id2label = create_label_mapping(GEO_TYPES)
            encoded_batch = encode_batch(final_batch, tokenizer, label2id, MAX_LENGTH)
            
            dataset_info = {
                'label2id': label2id,
                'id2label': id2label,
                'num_labels': len(label2id),
                'entity_types': GEO_TYPES,
                'tokenizer_name': MODEL_NAME,
                'max_length': MAX_LENGTH
            }
            
            prepared_metadata = {
                "source_batch": raw_path.name,
                "batch_id": final_batch[0].get('batch_id', 0),
                "num_samples": len(encoded_batch)
            }
            prepared_storage.save_batch(encoded_batch, prepared_metadata, dataset_info)
            all_encoded_batches.append(encoded_batch)
        
        if batch_num >= 2:
            break
    
    print("\n3. Обучение модели...")
    latest_prepared = prepared_storage.get_latest_batch()
    if latest_prepared:
        model_path, version = train_from_prepared_batch(
            prepared_storage=prepared_storage,
            batch_path=latest_prepared,
            model_name=MODEL_NAME,
            models_dir="./data/models",
            num_epochs=1,
            batch_size=4
        )
        
        registry = ModelRegistry()
        registry.register_model(
            model_path=model_path,
            metrics=None,
            metadata={'status': 'trained', 'source_batch': latest_prepared.name}
        )
        
        print("\n4. Валидация модели...")
        metrics = validate_model(
            model_path=Path(model_path),
            prepared_storage=prepared_storage,
            batch_path=latest_prepared,
            device="cpu"
        )
        registry.update_metrics(version, metrics)
        
        print(f"\nОбучение завершено!")
        print(f"   Версия модели: {version}")
        print(f"   F1: {metrics['f1']:.4f}")
    
    print("\n" + "="*60)
    print("UPDATE ПАЙПЛАЙН ЗАВЕРШЁН")
    print("="*60)
    return True

def run_inference(file_path: str):
    """
    Применение модели к данным из файла.
    
    Args:
        file_path: Путь к файлу с текстами (построчно)
    """
    print("\n" + "="*60)
    print("ЗАПУСК INFERENCE")
    print("="*60)
    
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Ошибка: файл {file_path} не найден")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Загружено текстов: {len(texts)}")
    
    pipeline = InferencePipeline(
        model_version=None,
        tokenizer_name=MODEL_NAME,
        device="cpu",
        max_length=MAX_LENGTH
    )
    
    results = []
    for i, text in enumerate(texts):
        entities = pipeline.predict(text)
        results.append({
            'text': text,
            'entities': entities,
            'index': i
        })
        print(f"\n{i+1}. {text[:50]}...")
        if entities:
            for e in entities:
                print(f"     {e['type']}: '{e['text']}'")
        else:
            print("     Сущности не найдены")
    
    output_path = Path(f"data/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        results_serializable = _make_serializable(results)
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\nРезультаты сохранены в: {output_path}")
    return str(output_path)


def run_summary():
    """
    Генерация отчёта о работе системы.
    """
    print("\n" + "="*60)
    print("ГЕНЕРАЦИЯ ОТЧЁТА")
    print("="*60)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'stages': {}
    }
    
    raw_storage = RawDataStorage("data/raw")
    raw_batches = raw_storage.list_batches()
    report['stages']['raw'] = {
        'num_batches': len(raw_batches),
        'batches': [b.name for b in raw_batches]
    }
    if raw_batches:
        latest_raw = raw_storage.get_metadata(raw_batches[-1])
        report['stages']['raw']['latest_stats'] = {
            'num_documents': latest_raw.get('num_documents'),
            'total_entities': latest_raw.get('total_entities'),
            'entity_types': list(latest_raw.get('entity_type_distribution', {}).keys())[:10]
        }
    
    processed_storage = ProcessedDataStorage("data/processed")
    processed_batches = processed_storage.list_batches()
    report['stages']['processed'] = {
        'num_batches': len(processed_batches),
        'batches': [b.name for b in processed_batches]
    }
    if processed_batches:
        latest_processed = processed_storage.get_metadata(processed_batches[-1])
        report['stages']['processed']['latest_stats'] = {
            'num_documents': latest_processed.get('num_documents'),
            'consistency_ratio': latest_processed.get('consistency_ratio'),
            'docs_removed': latest_processed.get('docs_removed')
        }
    
    prepared_storage = PreparedDataStorage("data/prepared")
    prepared_batches = prepared_storage.list_batches()
    report['stages']['prepared'] = {
        'num_batches': len(prepared_batches),
        'batches': [b.name for b in prepared_batches]
    }
    if prepared_batches:
        latest_prepared = prepared_storage.load_batch(prepared_batches[-1])
        report['stages']['prepared']['latest_stats'] = {
            'num_samples': latest_prepared['metadata'].get('num_samples'),
            'max_length': latest_prepared['metadata'].get('max_length')
        }
    
    registry = ModelRegistry()
    latest_model = registry.get_latest()
    if latest_model:
        report['models'] = {
            'latest_version': latest_model['version'],
            'latest_metrics': latest_model.get('metrics', {}),
            'all_versions': registry.list_versions()
        }
    else:
        report['models'] = {'status': 'no_models_trained'}
    
    report_path = Path(f"reports/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\nОТЧЁТ:")
    print(f"   Время: {report['timestamp']}")
    print(f"   Raw батчей: {report['stages']['raw']['num_batches']}")
    print(f"   Processed батчей: {report['stages']['processed']['num_batches']}")
    print(f"   Prepared батчей: {report['stages']['prepared']['num_batches']}")
    
    if 'models' in report and report['models'].get('latest_version'):
        print(f"   Последняя модель: {report['models']['latest_version']}")
        metrics = report['models']['latest_metrics']
        if metrics:
            print(f"     F1: {metrics.get('f1', 'N/A')}")
            print(f"     Precision: {metrics.get('precision', 'N/A')}")
            print(f"     Recall: {metrics.get('recall', 'N/A')}")
    
    print(f"\nОтчёт сохранён в: {report_path}")
    return str(report_path)

def main():
    parser = argparse.ArgumentParser(description="NER MLOps Pipeline")
    parser.add_argument(
        "-mode", 
        required=True, 
        choices=["inference", "update", "summary"],
        help="Режим работы: update (обучение), inference (предсказание), summary (отчёт)"
    )
    parser.add_argument(
        "-file", 
        help="Путь к файлу для инференса (требуется для mode=inference)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "update":
        success = run_update()
        print(success)
        
    elif args.mode == "inference":
        if not args.file:
            print("Ошибка: для режима inference требуется указать -file")
            sys.exit(1)
        result_path = run_inference(args.file)
        print(result_path)
        
    elif args.mode == "summary":
        report_path = run_summary()
        print(report_path)


if __name__ == "__main__":
    main()