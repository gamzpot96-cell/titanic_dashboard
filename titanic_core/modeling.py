"""학습 데이터에서만 전처리·모델 선택을 하고, 남겨 둔 데이터로 한 번 평가한다."""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, PROJECT_ROOT,
    RANDOM_STATE, RAW_FILE_NAME, prepare_directories,
)
from .data import file_sha256, load_dataset, make_features


def make_pipeline(classifier):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def train_models(root=PROJECT_ROOT):
    root = prepare_directories(root)
    frame = load_dataset(root)
    features = make_features(frame)[MODEL_FEATURES]
    target = frame["Survived"].astype(int)
    train_indices, test_indices = train_test_split(
        frame.index, test_size=0.2, stratify=target, random_state=RANDOM_STATE
    )
    x_train, x_test = features.loc[train_indices], features.loc[test_indices]
    y_train, y_test = target.loc[train_indices], target.loc[test_indices]
    candidates = {
        "logistic_regression": make_pipeline(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "random_forest": make_pipeline(RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=1,
        )),
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison = []
    for name, pipeline in candidates.items():
        scores = cross_val_score(pipeline, x_train, y_train, cv=folds, scoring="roc_auc", n_jobs=1)
        comparison.append({"model": name, "cv_roc_auc_mean": float(scores.mean()),
                           "cv_roc_auc_std": float(scores.std())})
    comparison.sort(key=lambda row: row["cv_roc_auc_mean"], reverse=True)
    selected_name = comparison[0]["model"]
    pipeline = candidates[selected_name]
    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, list(pipeline.classes_).index(1)]
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    baseline_accuracy = accuracy_score(y_test, baseline.predict(x_test))
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "baseline_accuracy": float(baseline_accuracy),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
    }
    test_predictions = pd.DataFrame({
        "PassengerId": frame.loc[test_indices, "PassengerId"].to_numpy(),
        "actual": y_test.to_numpy(), "predicted": predicted,
        "survival_probability": probabilities,
        "Sex": frame.loc[test_indices, "Sex"].to_numpy(),
        "Pclass": frame.loc[test_indices, "Pclass"].to_numpy(),
    })
    fairness_rows = []
    for column in ["Sex", "Pclass"]:
        for value, group in test_predictions.groupby(column):
            fairness_rows.append({
                "group_column": column, "group": str(value), "n": len(group),
                "accuracy": float(accuracy_score(group["actual"], group["predicted"])),
                "recall": float(recall_score(group["actual"], group["predicted"], zero_division=0)),
                "note": "소표본 지표. 이 값만으로 공정성 여부를 판정하지 않음.",
            })
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__}
    report = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected_name, "selection_metric": "training 5-fold ROC-AUC",
        "random_state": RANDOM_STATE, "train_rows": len(x_train), "test_rows": len(x_test),
        "train_passenger_ids": frame.loc[train_indices, "PassengerId"].tolist(),
        "test_passenger_ids": frame.loc[test_indices, "PassengerId"].tolist(),
        "features": MODEL_FEATURES, "metrics": metrics, "cv_comparison": comparison,
        "versions": versions, "raw_sha256": file_sha256(root / "data" / RAW_FILE_NAME),
        "scope": "정답이 있는 CSV 표본의 무작위 80/20 분할 평가. Kaggle 공식 test.csv 점수가 아님.",
        "limitations": [
            "모델은 학습용 80%로만 학습한 상태로 저장함. 평가 지표와 저장 모델이 일치함.",
            "동반 가족이 양쪽에 나뉠 수 있어 다른 표본에 대한 성능을 보장하지 않음.",
            "출력 확률은 보정하지 않은 모델 추정값. 실제 개인의 생존 가능성이나 인과관계가 아님.",
            "성별·객실 등급의 차이는 역사적 표본의 관찰 결과이며 가치 판단의 기준이 아님.",
        ],
    }
    model_path = root / "model"
    joblib.dump({"pipeline": pipeline, "features": MODEL_FEATURES,
                 "versions": versions, "selected_model": selected_name,
                 "raw_sha256": report["raw_sha256"]}, model_path / "titanic_survival_model.pkl")
    (model_path / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (model_path / "classification_report.txt").write_text(
        classification_report(y_test, predicted, target_names=["비생존", "생존"], digits=4, zero_division=0), encoding="utf-8"
    )
    test_predictions.to_csv(model_path / "holdout_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fairness_rows).to_csv(model_path / "group_metrics.csv", index=False, encoding="utf-8-sig")
    return report


def load_model(root=PROJECT_ROOT):
    path = Path(root) / "model" / "titanic_survival_model.pkl"
    if not path.exists():
        raise FileNotFoundError("먼저 06_model_training.ipynb 또는 python prepare_project.py를 실행하세요.")
    # 이 프로젝트가 직접 만든 파일만 사용한다. 외부에서 받은 임의의 pkl은 열지 않는다.
    bundle = joblib.load(path)
    if bundle["versions"]["scikit-learn"] != sklearn.__version__:
        raise RuntimeError("모델을 만든 scikit-learn 버전과 다릅니다. 현재 환경에서 다시 학습하세요.")
    if bundle["raw_sha256"] != file_sha256(Path(root) / "data" / RAW_FILE_NAME):
        raise RuntimeError("현재 데이터와 학습한 데이터가 다릅니다. 모델을 다시 학습하세요.")
    return bundle


def predict_passenger(pclass, sex, age, sibsp, parch, fare, embarked, root=PROJECT_ROOT):
    if pclass not in [1, 2, 3] or sex not in ["female", "male"] or embarked not in ["C", "Q", "S"]:
        raise ValueError("객실 등급·성별·승선 항구를 확인하세요.")
    for name, value, upper in [("나이", age, 100), ("동반 형제·배우자", sibsp, 20),
                               ("동반 부모·자녀", parch, 20), ("요금", fare, 10000)]:
        if value is None or not np.isfinite(value) or not 0 <= value <= upper:
            raise ValueError(f"{name} 입력 범위를 확인하세요.")
    if int(sibsp) != sibsp or int(parch) != parch:
        raise ValueError("동반 가족 수는 정수로 입력하세요.")
    bundle = load_model(root)
    frame = pd.DataFrame([{"Pclass": int(pclass), "Sex": sex, "Age": float(age),
                           "SibSp": int(sibsp), "Parch": int(parch), "Fare": float(fare), "Embarked": embarked}])
    pipeline = bundle["pipeline"]
    probability = float(pipeline.predict_proba(make_features(frame)[MODEL_FEATURES])[0, list(pipeline.classes_).index(1)])
    return {"prediction": int(probability >= 0.5), "survival_probability": probability,
            "selected_model": bundle["selected_model"]}
