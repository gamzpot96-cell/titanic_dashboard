"""원본을 보존하고, 수집 이력과 분석용 복사본을 만든다."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DATASET_HANDLE, PROJECT_ROOT, RAW_FEATURES, RAW_FILE_NAME, SOURCE_URL,
    prepare_directories,
)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_dataset(frame):
    required = ["PassengerId", "Survived"] + RAW_FEATURES
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"필수 열이 없습니다: {missing}")
    if frame.empty or frame["PassengerId"].isna().any():
        raise ValueError("빈 데이터이거나 승객 번호가 누락되어 있습니다.")
    if frame["PassengerId"].duplicated().any():
        raise ValueError("승객 번호가 중복되어 있습니다. 원본을 확인하세요.")
    if frame["Survived"].isna().any() or not set(frame["Survived"].unique()) <= {0, 1}:
        raise ValueError("Survived 열은 누락 없이 0 또는 1이어야 합니다.")
    if set(frame["Survived"].unique()) != {0, 1}:
        raise ValueError("생존·비생존 두 종류의 정답이 필요합니다.")
    if not frame["Pclass"].isin([1, 2, 3]).all():
        raise ValueError("Pclass는 1, 2, 3이어야 합니다.")
    if not frame["Sex"].isin(["female", "male"]).all():
        raise ValueError("Sex 값이 데이터 설명과 다릅니다.")
    if not frame["Embarked"].dropna().isin(["C", "Q", "S"]).all():
        raise ValueError("Embarked 값이 데이터 설명과 다릅니다.")
    for name in ["Age", "SibSp", "Parch", "Fare"]:
        values = pd.to_numeric(frame[name], errors="raise").dropna()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name}에 유효하지 않은 숫자가 있습니다.")
    return frame


def download_dataset(root=PROJECT_ROOT):
    """공개 데이터 버전 1을 kagglehub로 받는다. 기존 원본은 덮어쓰지 않는다."""
    root = prepare_directories(root)
    destination = root / "data" / RAW_FILE_NAME
    manifest_path = root / "data" / "collection_manifest.json"
    if destination.exists():
        frame = validate_dataset(pd.read_csv(destination))
        if not manifest_path.exists():
            raise ValueError("원본은 있지만 수집 기록이 없습니다. 출처 확인 후 별도 경로에서 수집하세요.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["sha256"] != file_sha256(destination):
            raise ValueError("수집 당시와 원본 파일이 다릅니다. 원본을 확인하세요.")
        return frame

    import kagglehub
    try:
        cache_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    except Exception as error:
        raise RuntimeError(
            "Kaggle 데이터 수집에 실패했습니다. 인터넷 연결을 확인하고 다시 실행하세요. "
            "인증이 요구되는 경우 본인의 Kaggle 계정을 사용하세요. "
            "다른 데이터로 자동 대체하지 않았습니다."
        ) from error
    candidates = list(cache_path.rglob(RAW_FILE_NAME))
    if len(candidates) != 1:
        raise FileNotFoundError(f"{RAW_FILE_NAME} 파일을 한 개 찾을 수 있어야 합니다: {cache_path}")
    source_path = candidates[0]
    frame = validate_dataset(pd.read_csv(source_path))
    shutil.copy2(source_path, destination)
    manifest = {
        "dataset_handle": DATASET_HANDLE,
        "source_url": SOURCE_URL,
        "download_method": "kagglehub.dataset_download",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file": RAW_FILE_NAME,
        "sha256": file_sha256(destination),
        "rows": len(frame), "columns": list(frame.columns),
        "license_at_collection": "CC0: Public Domain (Kaggle dataset metadata)",
        "scope": "이 CSV에 정답이 있는 승객 표본. 전체 탑승객·승무원 전체가 아님.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return frame


def load_dataset(root=PROJECT_ROOT):
    path = Path(root) / "data" / RAW_FILE_NAME
    manifest_path = path.parent / "collection_manifest.json"
    if not path.exists():
        raise FileNotFoundError("먼저 02_data_loading.ipynb 또는 python prepare_project.py를 실행하세요.")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if file_sha256(path) != manifest["sha256"]:
            raise ValueError("수집한 원본과 현재 파일의 검증값이 다릅니다.")
    return validate_dataset(pd.read_csv(path))


def make_features(frame):
    """행별 계산만 한다. 중앙값·평균값을 여기서 학습하지 않는다."""
    result = frame[RAW_FEATURES].copy()
    result["FamilySize"] = result["SibSp"] + result["Parch"] + 1
    result["IsAlone"] = (result["FamilySize"] == 1).astype(int)
    return result


def save_prepared_data(root=PROJECT_ROOT):
    frame = load_dataset(root)
    prepared = make_features(frame)
    prepared.insert(0, "PassengerId", frame["PassengerId"])
    prepared["Survived"] = frame["Survived"]
    # 이름·티켓·선실번호는 분석용 복사본에 넣지 않는다. 결측값은 모델 안에서 처리한다.
    path = Path(root) / "data" / "titanic_prepared.csv"
    prepared.to_csv(path, index=False, encoding="utf-8-sig")
    return prepared


def grouped_survival(frame, column):
    table = frame.groupby(column, dropna=False, observed=False)["Survived"].agg(
        passengers="size", survivors="sum"
    ).reset_index()
    table["survival_rate_pct"] = table["survivors"] / table["passengers"] * 100
    return table
