"""실행 위치: 이 파일이 있는 폴더. 명령: python prepare_project.py"""
import argparse

from titanic_core.config import PROJECT_ROOT
from titanic_core.data import download_dataset, save_prepared_data
from titanic_core.modeling import train_models
from titanic_core.evidence import verify_facts


def main():
    parser = argparse.ArgumentParser(description="타이타닉 분석용 데이터와 모델을 준비합니다.")
    parser.add_argument("--verify-facts", action="store_true", help="저장된 사실표를 읽기 전용으로 검증합니다.")
    args = parser.parse_args()
    facts = verify_facts(PROJECT_ROOT)
    print(f"고정 사실표 검증 완료: {len(facts)}개 근거가 저장본과 일치합니다.")
    if args.verify_facts:
        return

    frame = download_dataset(PROJECT_ROOT)
    print(f"원본 확인: {len(frame):,}행")
    save_prepared_data(PROJECT_ROOT)
    report = train_models(PROJECT_ROOT)
    print(f"모델: {report['selected_model']}")
    print(f"평가 정확도: {report['metrics']['accuracy']:.2%}")
    print("준비 완료. python -m streamlit run app.py로 실행합니다.")


if __name__ == "__main__":
    main()
