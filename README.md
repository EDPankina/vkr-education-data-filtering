# Алгоритм фильтрации образовательных данных

Репозиторий содержит программный прототип, разработанный в рамках выпускной квалификационной работы.

## Тема работы

Разработка алгоритма фильтрации данных при оценке объективности в исследованиях качества образования.

## Назначение прототипа

Прототип предназначен для предварительного выявления объектов с признаками потенциальной необъективности в результатах массовых оценочных процедур.

Алгоритм анализирует не отдельные баллы участников, а распределение результатов внутри группы: школы, класса или иной совокупности участников.

## Используемые признаки

Для каждого объекта рассчитываются:

- количество участников;
- средний балл;
- медиана;
- стандартное отклонение;
- минимальный и максимальный баллы;
- доля высоких результатов;
- доля низких результатов;
- доля результатов около пороговых значений;
- показатель скачка около порога;
- асимметрия;
- эксцесс;
- отклонение среднего балла объекта от среднего по выборке.

## Используемые модели

В работе сравниваются следующие модели машинного обучения:

- Logistic Regression;
- Random Forest;
- Gradient Boosting;
- Isolation Forest.

Наилучший результат показала модель Random Forest.

| Метрика | Значение |
|---|---:|
| Accuracy | 0.992 |
| Precision | 1.000 |
| Recall | 0.973 |
| F1-мера | 0.986 |

## Структура репозитория

```text
## Структура репозитория

```text
vkr-education-data-filtering/
├── main.py
├── requirements.txt
├── README.md
├── data/
│   └── synthetic_education_results.csv
├── results/
│   ├── features_dataset.csv
│   ├── model_metrics.csv
│   ├── final_predictions.csv
│   ├── filtered_dataset.csv
│   ├── suspicious_objects.csv
│   ├── real_features_profiles.csv
│   ├── real_profile_summary.csv
│   ├── real_suspicious_profiles.csv
│   ├── real_strong_stable_results.csv
│   ├── real_small_samples.csv
│   └── real_feature_importance.csv
└── images/
    ├── normal_distribution.png
    ├── suspicious_distribution.png
    ├── model_metrics_comparison.png
    ├── confusion_matrix.png
    └── real_feature_importance.png
```

## Запуск

Установить зависимости:

```bash
pip install -r requirements.txt
```

Запустить прототип:

```bash
python main.py
```

После запуска будут сформированы CSV-файлы и изображения в папках `data`, `results` и `images`.

## Автор

Панкина Елизавета, группа 406.
