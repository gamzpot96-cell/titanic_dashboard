"""실행 위치: 이 파일이 있는 폴더. 명령: python prepare_project.py"""
from titanic_core.config import PROJECT_ROOT
from titanic_core.data import download_dataset, save_prepared_data
from titanic_core.modeling import train_models
from titanic_core.evidence import create_evidence


def main():
    frame = download_dataset(PROJECT_ROOT)
    print(f"원본 확인: {len(frame):,}행")
    save_prepared_data(PROJECT_ROOT)
    report = train_models(PROJECT_ROOT)
    facts = create_evidence(PROJECT_ROOT)
    print(f"모델: {report['selected_model']}")
    print(f"평가 정확도: {report['metrics']['accuracy']:.2%}")
    print(f"사실표: {len(facts)}개 근거")
    print("준비 완료. python -m streamlit run app.py로 실행합니다.")


if __name__ == "__main__":
    main()
