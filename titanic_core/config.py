from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_HANDLE = "yasserh/titanic-dataset/versions/1"
SOURCE_URL = "https://www.kaggle.com/datasets/yasserh/titanic-dataset"
RAW_FILE_NAME = "Titanic-Dataset.csv"
RAW_FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42


def prepare_directories(root=PROJECT_ROOT):
    root = Path(root)
    for name in ["data", "model", "rag"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root
