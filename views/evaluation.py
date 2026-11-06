import json
import pandas as pd
import streamlit as st
from titanic_core.config import PROJECT_ROOT

st.title("모델 평가")
path = PROJECT_ROOT / "model" / "metrics.json"
if not path.exists():
    st.error("평가 파일이 없습니다. python prepare_project.py로 모델을 준비하세요.")
    st.stop()
report = json.loads(path.read_text(encoding="utf-8"))
st.write(f"선택 모델: {report['selected_model']} · 학습 {report['train_rows']}명 / 평가 {report['test_rows']}명")
metrics = report["metrics"]
cols = st.columns(3)
cols[0].metric("평가 정확도", f"{metrics['accuracy'] * 100:.2f}%")
cols[1].metric("생존 재현율", f"{metrics['recall'] * 100:.2f}%")
cols[2].metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
st.caption(f"기준 모델 정확도: {metrics['baseline_accuracy'] * 100:.2f}% · 공식 Kaggle 대회 점수가 아닙니다.")
st.write("학습 표본 안에서 5분할 교차 검증 ROC-AUC가 높은 모델을 선택했습니다.")
st.dataframe(pd.DataFrame(report["cv_comparison"]), hide_index=True)
st.subheader("혼동 행렬")
st.caption("행은 실제 결과, 열은 예측 결과입니다. 숫자는 인원(명)입니다.")
st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=["실제 비생존", "실제 생존"], columns=["예측 비생존", "예측 생존"]))
st.subheader("하위 집단 평가")
st.dataframe(pd.read_csv(PROJECT_ROOT / "model" / "group_metrics.csv"), hide_index=True)
st.caption("각 집단의 표본 수와 오류를 함께 보세요. 이 지표만으로 공정성을 판정하지 않습니다.")
with st.expander("분석 한계와 학습 환경"):
    for limitation in report["limitations"]:
        st.write("• " + limitation)
    st.json(report["versions"])
