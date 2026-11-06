"""실행: 이 폴더에서 python -m streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="타이타닉 데이터 분석", page_icon="🚢", layout="wide")
page = st.navigation({
    "데이터": [st.Page("views/home.py", title="프로젝트 안내", default=True),
               st.Page("views/analysis.py", title="데이터 분석")],
    "모델": [st.Page("views/predict.py", title="생존 예측 실습"),
             st.Page("views/evaluation.py", title="모델 평가")],
    "근거": [st.Page("views/qa.py", title="분석 Q&A")],
})
st.sidebar.caption("표본 데이터 기반 학습용 프로젝트")
page.run()
