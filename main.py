import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

np.random.seed(42)


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


def create_raw_data(n_objects=500):
    records = []
    subjects = ["Математика", "Русский язык", "Обществознание"]

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
                "anomaly_type": anomaly_type,
            })

    return pd.DataFrame(records)


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
    return pd.Series({
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
        "anomaly_type": group["anomaly_type"].iloc[0],
    })


def build_features(raw_data):
    features_data = raw_data.groupby("object_id").apply(calculate_features).reset_index(drop=True)
    features_data["mean_deviation"] = features_data["mean_score"] - features_data["mean_score"].mean()
    return features_data


def plot_distributions(raw_data):
    normal_id = raw_data[raw_data["is_suspicious"] == 0]["object_id"].iloc[0]
    suspicious_id = raw_data[raw_data["is_suspicious"] == 1]["object_id"].iloc[0]
    normal_scores = raw_data[raw_data["object_id"] == normal_id]["score"]
    suspicious_scores = raw_data[raw_data["object_id"] == suspicious_id]["score"]

    plt.figure(figsize=(10, 6))
    plt.hist(normal_scores, bins=10)
    plt.title("Распределение баллов для условно объективного объекта")
    plt.xlabel("Балл")
    plt.ylabel("Количество участников")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/normal_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(suspicious_scores, bins=10)
    plt.title("Распределение баллов для объекта с признаками искажения")
    plt.xlabel("Балл")
    plt.ylabel("Количество участников")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/suspicious_distribution.png", dpi=300)
    plt.close()


def train_and_compare_models(features_data):
    feature_columns = [
        "n_students", "mean_score", "median_score", "std_score", "min_score", "max_score",
        "high_score_share", "low_score_share", "threshold_share", "threshold_jump",
        "skewness", "kurtosis", "mean_deviation"
    ]
    X = features_data[feature_columns]
    y = features_data["is_suspicious"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    models = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))]),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Isolation Forest": Pipeline([("scaler", StandardScaler()), ("model", IsolationForest(contamination=0.3, random_state=42))]),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        if name == "Isolation Forest":
            raw_pred = model.predict(X_test)
            y_pred = np.where(raw_pred == -1, 1, 0)
        else:
            y_pred = model.predict(X_test)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv("results/model_metrics.csv", index=False, encoding="utf-8-sig")

    metrics_plot = metrics_df.set_index("model")[["accuracy", "precision", "recall", "f1"]]
    ax = metrics_plot.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Сравнение качества моделей")
    ax.set_ylabel("Значение метрики")
    ax.set_xlabel("Модель")
    ax.grid(axis="y")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("images/model_metrics_comparison.png", dpi=300)
    plt.close()

    final_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    final_model.fit(X_train, y_train)
    final_pred = final_model.predict(X_test)
    cm = confusion_matrix(y_test, final_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Объективные", "Подозрительные"])
    disp.plot()
    plt.title("Матрица ошибок: Random Forest")
    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png", dpi=300)
    plt.close()

    final_model.fit(X, y)
    features_data["predicted_label"] = final_model.predict(X)
    features_data["risk_probability"] = final_model.predict_proba(X)[:, 1]
    features_data.to_csv("results/final_predictions.csv", index=False, encoding="utf-8-sig")
    features_data[features_data["predicted_label"] == 0].to_csv("results/filtered_dataset.csv", index=False, encoding="utf-8-sig")
    features_data[features_data["predicted_label"] == 1].to_csv("results/suspicious_objects.csv", index=False, encoding="utf-8-sig")
    return metrics_df


def main():
    raw_data = create_raw_data()
    raw_data.to_csv("data/synthetic_education_results.csv", index=False, encoding="utf-8-sig")
    features_data = build_features(raw_data)
    features_data.to_csv("results/features_dataset.csv", index=False, encoding="utf-8-sig")
    plot_distributions(raw_data)
    metrics_df = train_and_compare_models(features_data)
    print("Готово. Файлы сохранены в папках data, results и images.")
    print(metrics_df)


if __name__ == "__main__":
    main()
