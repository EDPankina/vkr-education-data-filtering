# -*- coding: utf-8 -*-
"""Блок 1. Импорт библиотек
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from scipy.stats import skew, kurtosis

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

np.random.seed(42)

"""Блок 2. Генерация синтетических данных"""

def generate_normal_scores(n_students, mean, std, min_score=0, max_score=100):
    scores = np.random.normal(mean, std, n_students)
    scores = np.clip(scores, min_score, max_score)
    return np.round(scores).astype(int)


def generate_suspicious_scores(n_students, anomaly_type, min_score=0, max_score=100):
    if anomaly_type == "high_scores":
        scores = np.random.normal(82, 7, n_students)

    elif anomaly_type == "low_variance":
        scores = np.random.normal(75, 2, n_students)

    elif anomaly_type == "threshold_shift":
        # Концентрация около порога, например 60 баллов
        part1 = np.random.normal(61, 2, int(n_students * 0.6))
        part2 = np.random.normal(75, 8, n_students - len(part1))
        scores = np.concatenate([part1, part2])

    elif anomaly_type == "mixed":
        part1 = np.random.normal(85, 5, int(n_students * 0.5))
        part2 = np.random.normal(61, 2, n_students - len(part1))
        scores = np.concatenate([part1, part2])

    else:
        scores = np.random.normal(70, 10, n_students)

    scores = np.clip(scores, min_score, max_score)
    return np.round(scores).astype(int)


records = []

n_objects = 500
subjects = ["Математика", "Русский язык", "Обществознание"]
thresholds = [40, 60, 80]

for object_id in range(1, n_objects + 1):
    n_students = np.random.randint(15, 36)
    subject = np.random.choice(subjects)
    year = 2026

    is_suspicious = np.random.choice([0, 1], p=[0.7, 0.3])

    if is_suspicious == 0:
        mean = np.random.uniform(50, 75)
        std = np.random.uniform(8, 15)
        scores = generate_normal_scores(n_students, mean, std)
        anomaly_type = "normal"
    else:
        anomaly_type = np.random.choice(["high_scores", "low_variance", "threshold_shift", "mixed"])
        scores = generate_suspicious_scores(n_students, anomaly_type)

    for student_num, score in enumerate(scores, start=1):
        records.append({
            "object_id": object_id,
            "student_id": f"{object_id}_{student_num}",
            "subject": subject,
            "year": year,
            "score": score,
            "is_suspicious": is_suspicious,
            "anomaly_type": anomaly_type
        })

raw_data = pd.DataFrame(records)

raw_data.head()

"""Блок 3. Сохранение исходного датасета"""

raw_data.to_csv("synthetic_education_results.csv", index=False, encoding="utf-8-sig")

print("Размер исходного набора данных:", raw_data.shape)
print("Количество объектов:", raw_data["object_id"].nunique())
print("Количество записей по классам:")
print(raw_data["is_suspicious"].value_counts())

"""Блок 4. Расчет признаков"""

def threshold_concentration(scores, thresholds=[40, 60, 80], window=2):
    scores = np.array(scores)
    count = 0

    for threshold in thresholds:
        count += np.sum((scores >= threshold - window) & (scores <= threshold + window))

    return count / len(scores)


def threshold_jump(scores, threshold=60):
    scores = np.array(scores)
    below = np.sum((scores >= threshold - 3) & (scores < threshold))
    above = np.sum((scores >= threshold) & (scores <= threshold + 3))

    return (above - below) / len(scores)


def calculate_features(group):
    scores = group["score"].values

    features = {
        "object_id": group["object_id"].iloc[0],
        "subject": group["subject"].iloc[0],
        "year": group["year"].iloc[0],
        "n_students": len(scores),
        "mean_score": np.mean(scores),
        "median_score": np.median(scores),
        "std_score": np.std(scores),
        "min_score": np.min(scores),
        "max_score": np.max(scores),
        "high_score_share": np.mean(scores >= 80),
        "low_score_share": np.mean(scores < 40),
        "threshold_share": threshold_concentration(scores),
        "threshold_jump": threshold_jump(scores),
        "skewness": skew(scores),
        "kurtosis": kurtosis(scores),
        "is_suspicious": group["is_suspicious"].iloc[0],
        "anomaly_type": group["anomaly_type"].iloc[0]
    }

    return pd.Series(features)


features_data = raw_data.groupby("object_id").apply(calculate_features).reset_index(drop=True)

features_data["mean_deviation"] = features_data["mean_score"] - features_data["mean_score"].mean()

features_data.to_csv("features_dataset.csv", index=False, encoding="utf-8-sig")

features_data.head()

"""Блок 5. Графики распределений"""

normal_object = features_data[features_data["is_suspicious"] == 0]["object_id"].iloc[0]
suspicious_object = features_data[features_data["is_suspicious"] == 1]["object_id"].iloc[0]

normal_scores = raw_data[raw_data["object_id"] == normal_object]["score"]
suspicious_scores = raw_data[raw_data["object_id"] == suspicious_object]["score"]

plt.figure(figsize=(8, 5))
plt.hist(normal_scores, bins=10)
plt.title("Распределение баллов для условно объективного объекта")
plt.xlabel("Балл")
plt.ylabel("Количество участников")
plt.grid(True)
plt.savefig("normal_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(suspicious_scores, bins=10)
plt.title("Распределение баллов для объекта с признаками искажения")
plt.xlabel("Балл")
plt.ylabel("Количество участников")
plt.grid(True)
plt.savefig("suspicious_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

"""Блок 6. Обучение моделей"""

feature_columns = [
    "n_students",
    "mean_score",
    "median_score",
    "std_score",
    "min_score",
    "max_score",
    "high_score_share",
    "low_score_share",
    "threshold_share",
    "threshold_jump",
    "skewness",
    "kurtosis",
    "mean_deviation"
]

X = features_data[feature_columns]
y = features_data["is_suspicious"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
results_df

"""Блок 7. Isolation Forest"""

iso_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", IsolationForest(
        contamination=0.3,
        random_state=42
    ))
])

iso_pipeline.fit(X_train)

iso_pred_raw = iso_pipeline.predict(X_test)

# Isolation Forest возвращает: -1 для аномалий, 1 для нормальных объектов
iso_pred = np.where(iso_pred_raw == -1, 1, 0)

iso_result = {
    "model": "Isolation Forest",
    "accuracy": accuracy_score(y_test, iso_pred),
    "precision": precision_score(y_test, iso_pred),
    "recall": recall_score(y_test, iso_pred),
    "f1": f1_score(y_test, iso_pred)
}

results_df = pd.concat([results_df, pd.DataFrame([iso_result])], ignore_index=True)
results_df

"""Блок 8. Таблица метрик и график"""

results_df.to_csv("model_metrics.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(9, 5))
x = np.arange(len(results_df["model"]))
width = 0.2

plt.bar(x - 1.5 * width, results_df["accuracy"], width, label="Accuracy")
plt.bar(x - 0.5 * width, results_df["precision"], width, label="Precision")
plt.bar(x + 0.5 * width, results_df["recall"], width, label="Recall")
plt.bar(x + 1.5 * width, results_df["f1"], width, label="F1")

plt.xticks(x, results_df["model"], rotation=20)
plt.ylim(0, 1.05)
plt.title("Сравнение качества моделей")
plt.ylabel("Значение метрики")
plt.legend()
plt.grid(axis="y")

plt.savefig("model_metrics_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

results_df

"""Блок 9. Матрица ошибок для лучшей модели"""

best_model_name = results_df.sort_values("f1", ascending=False).iloc[0]["model"]
best_model_name

if best_model_name in models:
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test)
else:
    y_pred_best = iso_pred

cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title(f"Матрица ошибок: {best_model_name}")
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.xticks([0, 1], ["Объективные", "Подозрительные"])
plt.yticks([0, 1], ["Объективные", "Подозрительные"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

print(classification_report(y_test, y_pred_best))

"""Блок 10. Формирование итоговой таблицы фильтрации"""

final_model = models["Random Forest"]
final_model.fit(X, y)

features_data["predicted_label"] = final_model.predict(X)

if hasattr(final_model, "predict_proba"):
    features_data["risk_probability"] = final_model.predict_proba(X)[:, 1]
else:
    features_data["risk_probability"] = features_data["predicted_label"]

filtered_data = features_data[features_data["predicted_label"] == 0].copy()
suspicious_data = features_data[features_data["predicted_label"] == 1].copy()

features_data.to_csv("final_predictions.csv", index=False, encoding="utf-8-sig")
filtered_data.to_csv("filtered_dataset.csv", index=False, encoding="utf-8-sig")
suspicious_data.to_csv("suspicious_objects.csv", index=False, encoding="utf-8-sig")

print("Всего объектов:", len(features_data))
print("Объектов, помеченных как подозрительные:", len(suspicious_data))
print("Объектов в очищенном наборе данных:", len(filtered_data))

features_data.sort_values("risk_probability", ascending=False).head(10)

"""### Блок 11. Апробация подхода на реальных данных

"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

real_file_path = "/content/pupils_ruma456_2019_2020.csv"
context_file_path = "/content/cont_data.xlsx"

# ---------- загрузка основного файла ----------
try:
    real_df = pd.read_csv(
        real_file_path,
        sep=",",
        encoding="utf-8-sig",
        low_memory=False
    )
except UnicodeDecodeError:
    real_df = pd.read_csv(
        real_file_path,
        sep=",",
        encoding="cp1251",
        low_memory=False
    )

real_df.columns = real_df.columns.str.strip()

# приводим возможные варианты названий к единым именам
real_df = real_df.rename(columns={
    "Код предмета (1- РУ, 2- МА)": "Код_предмета",
    "Код предмета": "Код_предмета",
    "ОО (логин школы)": "ЛогинОО",
    "Код ученика": "Код_ученика",
    "Year - Учебный год": "Year"
})

print("Размер реального набора данных:", real_df.shape)
print("Колонки основного файла:", real_df.columns.tolist())
display(real_df.head())

# ---------- загрузка контекстного файла ----------
context_df = pd.read_excel(context_file_path, sheet_name=0)
context_df.columns = context_df.columns.str.strip()

# в файле первая строка может дублировать заголовки, удаляем ее
context_df = context_df[context_df["Логин ОО"].astype(str) != "Логин ОО"].copy()

context_df = context_df.rename(columns={
    "Логин ОО": "ЛогинОО",
    "Где расположена образовательная организация?": "location_type",
    "Наименование населенного пункта": "settlement_name",
    "Размер населенного пункта": "settlement_size",
    "ОО является специальной (коррекционной)": "is_special_school",
    "Количество обучающихся в ОО (в сумме по всем классам)": "total_students_oo"
})

context_df["ЛогинОО"] = context_df["ЛогинОО"].astype(str).str.strip()
context_df["total_students_oo"] = pd.to_numeric(context_df["total_students_oo"], errors="coerce")

context_df["is_special_school"] = (
    context_df["is_special_school"]
    .astype(str)
    .str.lower()
    .str.strip()
    .map({"да": 1, "нет": 0})
)

print("Размер контекстного справочника:", context_df.shape)
print("Колонки контекстного файла:", context_df.columns.tolist())
display(context_df.head())

"""### Блок 12. Подготовка реальных данных"""

# ---------- базовая подготовка ----------
required_cols = ["Класс", "Код_предмета", "ЛогинОО", "Код_ученика", "Балл", "Отметка", "Year"]

missing_cols = [col for col in required_cols if col not in real_df.columns]
if missing_cols:
    raise ValueError(f"В основном файле отсутствуют колонки: {missing_cols}")

real_df["Балл"] = pd.to_numeric(real_df["Балл"], errors="coerce")
real_df["Отметка"] = pd.to_numeric(real_df["Отметка"], errors="coerce")
real_df["Класс"] = pd.to_numeric(real_df["Класс"], errors="coerce")
real_df["Код_предмета"] = pd.to_numeric(real_df["Код_предмета"], errors="coerce")
real_df["Year"] = pd.to_numeric(real_df["Year"], errors="coerce")

real_df = real_df.dropna(subset=["Балл", "Отметка", "Класс", "Код_предмета", "Year", "ЛогинОО"]).copy()

real_df["ЛогинОО"] = real_df["ЛогинОО"].astype(str).str.strip()

# код региона из логина школы: sch77171506 -> 77
real_df["Код_региона"] = (
    real_df["ЛогинОО"]
    .astype(str)
    .str.extract(r"(\d{2})")[0]
    .astype("Int64")
)

# максимальный балл по описанию базы
conditions = [
    (real_df["Код_предмета"] == 1) & (real_df["Класс"] == 4),
    (real_df["Код_предмета"] == 1) & (real_df["Класс"].isin([5, 6])),
    (real_df["Код_предмета"] == 2) & (real_df["Класс"].isin([4, 5, 6]))
]

choices = [38, 45, 20]

real_df["max_possible_score"] = np.select(conditions, choices, default=np.nan)
real_df = real_df.dropna(subset=["max_possible_score"]).copy()

# нормированный балл от 0 до 1
real_df["score_percent"] = real_df["Балл"] / real_df["max_possible_score"]

# добавляем контекст школы
real_df = real_df.merge(context_df, on="ЛогинОО", how="left")

# заполняем пропуски в контексте
real_df["location_type"] = real_df["location_type"].fillna("не указано")
real_df["settlement_size"] = real_df["settlement_size"].fillna("не указано")
real_df["settlement_name"] = real_df["settlement_name"].fillna("не указано")
real_df["is_special_school"] = real_df["is_special_school"].fillna(0)
real_df["total_students_oo"] = real_df["total_students_oo"].fillna(real_df["total_students_oo"].median())

print("После подготовки:", real_df.shape)
print("Количество школ:", real_df["ЛогинОО"].nunique())
print("Количество регионов:", real_df["Код_региона"].nunique())
print("Классы:", sorted(real_df["Класс"].dropna().unique()))
print("Предметы:", sorted(real_df["Код_предмета"].dropna().unique()))
print("Годы:", sorted(real_df["Year"].dropna().unique()))
display(real_df.head())

"""### Блок 13. Расчет порогов перехода между отметками"""

# Пороги перехода между отметками определяются по фактическим данным:
# минимальный балл, с которого начинается отметка 3, 4 и 5
thresholds = (
    real_df[real_df["Отметка"].isin([3, 4, 5])]
    .groupby(["Класс", "Код_предмета", "Year", "Отметка"])["Балл"]
    .min()
    .reset_index()
    .sort_values(["Класс", "Код_предмета", "Year", "Отметка"])
)

print("Пороги перехода между отметками:")
display(thresholds)

threshold_dict = {}

for _, row in thresholds.iterrows():
    key = (int(row["Класс"]), int(row["Код_предмета"]), int(row["Year"]))
    threshold_dict.setdefault(key, {})
    threshold_dict[key][int(row["Отметка"])] = int(row["Балл"])

"""Пороговые значения определяются по фактической шкале перевода баллов в отметки. Для каждой комбинации класса, предмета и года находится минимальный балл, соответствующий отметкам 3, 4 и 5. Далее около каждого такого порога рассчитывается скачок распределения.

### Блок 14. Расчет признаков на реальных данных
"""

group_cols = ["Код_региона", "ЛогинОО", "Класс", "Код_предмета", "Year"]

def calc_threshold_features(group):
    cls = int(group["Класс"].iloc[0])
    subj = int(group["Код_предмета"].iloc[0])
    year = int(group["Year"].iloc[0])
    key = (cls, subj, year)

    scores = group["Балл"].astype(int).values
    n = len(scores)

    result = {}

    for mark in [3, 4, 5]:
        threshold = threshold_dict.get(key, {}).get(mark)

        if threshold is None:
            result[f"jump_to_{mark}"] = 0
            result[f"share_near_{mark}"] = 0
            continue

        # два балла ниже порога и два балла выше/на пороге
        below = ((scores >= threshold - 2) & (scores < threshold)).sum()
        above = ((scores >= threshold) & (scores <= threshold + 1)).sum()
        near = ((scores >= threshold - 2) & (scores <= threshold + 1)).sum()

        # нормировка на размер группы
        result[f"jump_to_{mark}"] = (above - below) / n
        result[f"share_near_{mark}"] = near / n

    result["max_threshold_jump"] = max(
        result.get("jump_to_3", 0),
        result.get("jump_to_4", 0),
        result.get("jump_to_5", 0)
    )

    result["max_threshold_share"] = max(
        result.get("share_near_3", 0),
        result.get("share_near_4", 0),
        result.get("share_near_5", 0)
    )

    return result


def calc_real_features(group):
    scores = group["score_percent"].dropna().values
    raw_scores = group["Балл"].dropna().values
    n = len(scores)

    if n == 0:
        return pd.Series()

    threshold_features = calc_threshold_features(group)

    result = {
        "n_students": n,
        "mean_score": np.mean(scores),
        "median_score": np.median(scores),
        "std_score": np.std(scores),
        "min_score": np.min(scores),
        "max_score": np.max(scores),
        "score_range": np.max(scores) - np.min(scores),
        "high_score_share": np.mean(scores >= 0.8),
        "low_score_share": np.mean(scores <= 0.4),
        "skewness": skew(scores) if n > 2 else 0,
        "kurtosis": kurtosis(scores) if n > 3 else 0,
        "mean_raw_score": np.mean(raw_scores),
        "std_raw_score": np.std(raw_scores),
        "location_type": group["location_type"].iloc[0],
        "settlement_size": group["settlement_size"].iloc[0],
        "settlement_name": group["settlement_name"].iloc[0],
        "is_special_school": group["is_special_school"].iloc[0],
        "total_students_oo": group["total_students_oo"].iloc[0]
    }

    result.update(threshold_features)

    return pd.Series(result)


real_features = real_df.groupby(group_cols).apply(calc_real_features).reset_index()

print("Таблица признаков по реальным данным:", real_features.shape)
display(real_features.head())

"""### Блок 15. Контекстное сравнение школ"""

# Сравнение школы с объектами того же региона, класса, предмета и года
region_context_cols = ["Код_региона", "Класс", "Код_предмета", "Year"]

region_stats = (
    real_features
    .groupby(region_context_cols)["mean_score"]
    .agg(region_mean="mean", region_std="std", region_group_count="count")
    .reset_index()
)

real_features = real_features.merge(region_stats, on=region_context_cols, how="left")

real_features["z_mean_region"] = (
    (real_features["mean_score"] - real_features["region_mean"]) /
    real_features["region_std"].replace(0, np.nan)
).fillna(0)


# Более точное контекстное сравнение:
# регион + размер населенного пункта + класс + предмет + год
settlement_context_cols = ["Код_региона", "settlement_size", "Класс", "Код_предмета", "Year"]

settlement_stats = (
    real_features
    .groupby(settlement_context_cols)["mean_score"]
    .agg(context_mean="mean", context_std="std", context_group_count="count")
    .reset_index()
)

real_features = real_features.merge(settlement_stats, on=settlement_context_cols, how="left")

real_features["z_mean_context"] = (
    (real_features["mean_score"] - real_features["context_mean"]) /
    real_features["context_std"].replace(0, np.nan)
).fillna(0)

# если в контекстной группе слишком мало школ, используем региональное сравнение
real_features["z_mean_final"] = np.where(
    real_features["context_group_count"] >= 10,
    real_features["z_mean_context"],
    real_features["z_mean_region"]
)

print("Контекстное сравнение добавлено")
display(real_features[[
    "ЛогинОО",
    "Код_региона",
    "Класс",
    "Код_предмета",
    "Year",
    "settlement_size",
    "mean_score",
    "z_mean_region",
    "z_mean_context",
    "z_mean_final",
    "context_group_count"
]].head())

"""### Блок 16. Профилирование объектов"""

# Флаги признаков
real_features["flag_small_sample"] = real_features["n_students"] < 10

real_features["flag_high_mean"] = real_features["z_mean_final"] > 2

real_features["flag_low_std"] = (
    real_features["std_score"] <
    real_features["std_score"].quantile(0.10)
)

real_features["flag_high_min"] = (
    real_features["min_score"] >
    real_features["min_score"].quantile(0.90)
)

real_features["flag_high_share"] = (
    real_features["high_score_share"] >
    real_features["high_score_share"].quantile(0.90)
)

real_features["flag_threshold_jump"] = (
    real_features["max_threshold_jump"] >
    real_features["max_threshold_jump"].quantile(0.90)
)

real_features["flag_context_extreme"] = real_features["z_mean_final"] > 2.5

# риск-скор
# пороговый скачок весит сильнее, потому что это более содержательный признак риска,
# чем просто высокий средний балл
real_features["risk_score"] = (
    real_features["flag_low_std"].astype(int)
    + real_features["flag_high_min"].astype(int)
    + real_features["flag_high_share"].astype(int)
    + real_features["flag_threshold_jump"].astype(int) * 2
    + real_features["flag_context_extreme"].astype(int)
)

def define_profile(row):
    if row["flag_small_sample"]:
        return "малая выборка"

    # высокий результат без скачка и без сильного сжатия разброса
    if row["flag_high_mean"] and not row["flag_threshold_jump"] and not row["flag_low_std"]:
        return "высокий устойчивый результат"

    # риск - сочетание нескольких признаков, особенно если есть пороговый скачок
    if row["risk_score"] >= 3:
        return "профиль риска"

    return "без выраженных отклонений"

real_features["profile_type"] = real_features.apply(define_profile, axis=1)
real_features["risk_label"] = (real_features["profile_type"] == "профиль риска").astype(int)

profile_summary = (
    real_features["profile_type"]
    .value_counts()
    .reset_index()
)

profile_summary.columns = ["profile_type", "count"]

print("Сводка по профилям объектов:")
display(profile_summary)

display(real_features[[
    "ЛогинОО",
    "Код_региона",
    "Класс",
    "Код_предмета",
    "Year",
    "n_students",
    "settlement_size",
    "mean_score",
    "std_score",
    "max_threshold_jump",
    "z_mean_final",
    "risk_score",
    "profile_type"
]].head(20))

"""### Блок 17. Feature importance на реальных данных

На реальных данных отсутствует экспертная метка необъективности, поэтому Random Forest здесь используется не для доказательства точности, а для анализа того, какие признаки сильнее всего связаны с выделенным профилем риска.
"""

from sklearn.ensemble import RandomForestClassifier

numeric_features = [
    "n_students",
    "mean_score",
    "median_score",
    "std_score",
    "min_score",
    "max_score",
    "score_range",
    "high_score_share",
    "low_score_share",
    "jump_to_3",
    "jump_to_4",
    "jump_to_5",
    "max_threshold_jump",
    "share_near_3",
    "share_near_4",
    "share_near_5",
    "max_threshold_share",
    "skewness",
    "kurtosis",
    "z_mean_region",
    "z_mean_context",
    "z_mean_final",
    "total_students_oo",
    "is_special_school"
]

X_real = real_features[numeric_features].fillna(0)
y_real = real_features["risk_label"]

rf_real = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

rf_real.fit(X_real, y_real)

real_importance = (
    pd.DataFrame({
        "feature": numeric_features,
        "importance": rf_real.feature_importances_
    })
    .sort_values("importance", ascending=False)
)

print("Важность признаков для профиля риска на реальных данных:")
display(real_importance)

plt.figure(figsize=(10, 6))
plt.barh(real_importance["feature"].head(15), real_importance["importance"].head(15))
plt.gca().invert_yaxis()
plt.title("Важность признаков Random Forest для профиля риска")
plt.xlabel("Feature importance")
plt.tight_layout()
plt.savefig("real_feature_importance.png", dpi=300)
plt.show()

"""### Блок 18. Сохранение результатов"""

real_features.to_csv("real_features_profiles.csv", index=False, encoding="utf-8-sig")
profile_summary.to_csv("real_profile_summary.csv", index=False, encoding="utf-8-sig")
real_importance.to_csv("real_feature_importance.csv", index=False, encoding="utf-8-sig")

real_suspicious = real_features[real_features["profile_type"] == "профиль риска"].copy()
real_strong = real_features[real_features["profile_type"] == "высокий устойчивый результат"].copy()
real_small = real_features[real_features["profile_type"] == "малая выборка"].copy()

real_suspicious.to_csv("real_suspicious_profiles.csv", index=False, encoding="utf-8-sig")
real_strong.to_csv("real_strong_stable_results.csv", index=False, encoding="utf-8-sig")
real_small.to_csv("real_small_samples.csv", index=False, encoding="utf-8-sig")

print("Файлы сохранены:")
print("real_features_profiles.csv")
print("real_profile_summary.csv")
print("real_feature_importance.csv")
print("real_feature_importance.png")
print("real_suspicious_profiles.csv")
print("real_strong_stable_results.csv")
print("real_small_samples.csv")

"""### Блок 18. Ограничения реальной апробации

### Интерпретация результатов на реальных данных

На реальном наборе данных отсутствует экспертная целевая метка «объективный / необъективный объект». Поэтому результаты апробации не интерпретируются как доказательство нарушения.

Алгоритм выделяет статистические профили образовательных организаций:
- объекты без выраженных отклонений;
- объекты с высоким устойчивым результатом;
- объекты с профилем риска;
- объекты с малой выборкой.

Такой подход позволяет не смешивать сильные образовательные организации с объектами, у которых одновременно проявляются пороговые скачки, низкий разброс, высокая доля сильных результатов и другие признаки риска.

Для окончательной проверки необходима экспертная разметка, сопоставление с внешними списками проблемных объектов или анализ данных за несколько лет.
"""