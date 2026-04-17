# NER MLOps Pipeline
Проект автоматизации и визуализации решения задачи распознавания именованных сущностей 

Датасеты: NEREL

Модели: BERT

## Автор
Зимонин Михаил Юрьевич

@Hahahaha_mz

## Датасет
- **Название:** NEREL (iluvvatar/NEREL)
- **Источник:** Hugging Face
- **Задача:** Распознавание именованных сущностей (NER)
- **Целевые типы:** Географические сущности (COUNTRY, CITY, LOCATION, STATE_OR_PROV, DISTRICT)

## Модели
- **Базовая модель для дообучения:** BERT (DeepPavlov/rubert-base-cased)
---

## Структура проекта

.
├── config
├── data
│   ├── models
│   │   └── registry.json
│   ├── prepared
│   ├── processed
│   └── raw
├── logs
├── README.md
├── requirements.txt
├── run.py
├── src
│   ├── data_analysis
│   │   ├── __init__.py
│   │   ├── parser.py
│   │   ├── processed_data_storage.py
│   │   ├── quality_checker.py
│   │   ├── quality_filter.py
│   │   └── type_filter.py
│   ├── data_collection
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── metadata_calculator.py
│   │   └── storage.py
│   ├── data_preparation
│   │   ├── bio_encoder.py
│   │   ├── __init__.py
│   │   ├── prepared_data_storage.py
│   │   └── tokenizer_setup.py
│   ├── __init__.py
│   ├── serving
│   │   ├── inference_pipeline.py
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   └── predictor.py
│   ├── training
│   │   ├── __init__.py
│   │   ├── model_factory.py
│   │   └── trainer.py
│   ├── validation
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── model_registry.py
│   │   └── validator.py
│   └── visualization
│       └── __init__.py
├── tests
│   ├── test_data_analysis
│   │   └── parser_saver_test.py
│   ├── test_data_collection
│   │   ├── loader_saver_test.py
│   │   └── loader_test.py
│   ├── test_data_preparation
│   ├── test_full_pipeline.py
│   ├── test_inference
│   │   └── test_inf.py
│   ├── test_training
│   │   └── test_training.py
│   └── test_validation
│       └── test_val.py
└── test_texts.txt

---
## Этапы конвейера

| Этап | Название |
|------|----------|
| 1 | Сбор данных |
| 2 | Анализ данных |
| 3 | Подготовка данных |
| 4 | Обучение модели |
| 5 | Валидация модели |
| 6 | Обслуживание модели |

---

## Этап 1: Сбор данных (завершён)

### Что реализовано
- Загрузка датасета с Hugging Face
- Эмуляция потокового чтения (разбиение на батчи)
- Перемешивание данных (с возможностью установки seed для воспроизводимости)
- Сохранение сырых данных в `data/raw/`
- Расчёт базовых метапараметров

### Структура `data/raw/`
```
data/raw/
└── batch_20260323_141046_481/
    ├── metadata.json      # метаданные батча
    ├── doc_0000.json      # документ 1
    ├── doc_0001.json      # документ 2
    └── ...
```
```

---

## Этап 2: Анализ данных

### Что реализовано
- Парсинг сущностей из формата NEREL в структурированный вид (поддержка множественных интервалов)
- Фильтрация по географическим типам (COUNTRY, CITY, LOCATION, STATE_OR_PROV, DISTRICT)
- Проверка консистентности сущностей (сравнение текста по индексам)
- Фильтрация документов по порогам качества:
  - Минимальная доля консистентных сущностей (0.95)
  - Минимальное количество сущностей в документе (1)
- Сохранение обработанных данных в `data/processed/`
- Накопление метаданных (статистика фильтрации, качество)

### Структура `data/processed/`
```
data/processed/
└── batch_20260417_212124_602/
    ├── metadata.json      # метаданные с полной статистикой
    ├── doc_0000.json      # документ с parsed_entities и quality
    ├── doc_0001.json
    └── ...
```

---

## Этап 3: Подготовка данных

### Что реализовано
- Загрузка токенизатора BERT (DeepPavlov/rubert-base-cased)
- Токенизация текстов документов
- Генерация BIO-меток для каждого токена
- Преобразование в тензоры PyTorch с паддингом
- Сохранение подготовленных данных в `data/prepared/`

### Структура `data/prepared/`
```
data/prepared/
└── batch_20260417_214849_168/
    ├── tensors.pt         # тензоры input_ids, attention_mask, labels
    ├── metadata.json      # метаданные батча
    └── dataset_info.json  # label2id, id2label, num_labels
```

## Этап 4: Обучение модели

### Что реализовано
- Создание модели BERT для токеновой классификации (BertForTokenClassification)
- Автоматическое определение устройства (CPU/GPU)
- Настройка гиперпараметров обучения (эпохи, batch size, learning rate)
- Обучение на подготовленных данных из `data/prepared/`
- Сохранение обученной модели с версионированием в `data/models/`
- Регистрация модели с метаданными (версия, timestamp, источник данных)

### Структура `data/models/`
```
data/models/
├── v20260417_222927_383/
│   ├── config.json                # конфигурация модели
│   ├── model.safetensors          # веса модели
│   ├── training_metadata.json     # параметры обучения
│   └── ...
└── registry.json                  # реестр всех версий моделей
```

### Параметры обучения (по умолчанию)
| Параметр | Значение |
|----------|----------|
| Количество эпох | 1 |
| Размер батча | 4 |
| Скорость обучения | 2e-5 |
| Оптимизатор | AdamW |

---

## Этап 5: Валидация модели (завершён)

### Что реализовано
- Загрузка обученной модели из реестра версий
- Расчёт метрик качества NER (precision, recall, f1, accuracy)
- Валидация на подготовленных данных из `data/prepared/`
- Обновление метрик в реестре моделей
- Обработка numpy типов для JSON-сериализации

### Структура `src/validation/`
```
src/validation/
├── __init__.py
├── metrics.py           # расчёт метрик NER
├── validator.py         # класс для валидации модели
└── model_registry.py    # реестр версий моделей (общий для этапов 4-5)
```

### Метрики валидации
| Метрика | Описание |
|---------|----------|
| Accuracy | Доля правильно предсказанных токенов |
| Precision | Точность модели (доля правильных предсказаний среди всех предсказаний) |
| Recall | Полнота модели (доля найденных сущностей среди всех) |
| F1-score | Гармоническое среднее precision и recall |

## Этап 6: Обслуживание модели / Инференс (завершён)

### Что реализовано
- Загрузка модели из реестра по версии или последней
- Токенизация входного текста
- Предсказание сущностей с возвратом позиций в тексте
- Пакетная обработка нескольких текстов
- Сохранение результатов в JSON

### Модули `src/serving/`
| Модуль | Назначение |
|--------|------------|
| `model_loader.py` | Загрузка модели из реестра версий |
| `predictor.py` | Инференс на одном или нескольких текстах |
| `inference_pipeline.py` | Унифицированный пайплайн инференса |

---

### Режимы работы

Система поддерживает три режима работы через командную строку:

```bash
# 1. Обучение/дообучение модели
python run.py -mode update

# 2. Применение модели к новым данным (инференс)
python run.py -mode inference -file ./path_to_file.txt

# 3. Генерация отчёта о работе системы
python run.py -mode summary
```

### Возвращаемые значения

| Режим | Возвращает |
|-------|------------|
| `update` | `True` (успешно) или `False` (ошибка) |
| `inference` | Путь к JSON-файлу с предсказаниями |
| `summary` | Путь к JSON-файлу с отчётом |


## Установка

### Требования
- Python 3.9+
- Зависимости из `requirements.txt`
---

## Известные ограничения

- Версия `datasets` зафиксирована на 2.14.0 из-за формата NEREL (dataset scripts)
- Поддерживаются множественные интервалы сущностей (формат `EVENT 895 909;947 957`)

---

## Ссылки

- Датасет: https://huggingface.co/datasets/iluvvatar/NEREL
- Модель: https://huggingface.co/DeepPavlov/rubert-base-cased
