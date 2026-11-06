import pandas as pd
import plotly.express as px
import streamlit as st
from titanic_core.config import SOURCE_URL
from titanic_core.data import load_dataset, grouped_survival

st.title("데이터 분석")
try:
    data = load_dataset()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()
left, right = st.columns(2)
sexes = left.multiselect("기록된 성별", ["female", "male"], default=["female", "male"])
classes = right.multiselect("객실 등급", [1, 2, 3], default=[1, 2, 3])
filtered = data[data["Sex"].isin(sexes) & data["Pclass"].isin(classes)]
if filtered.empty:
    st.warning("조건에 해당하는 승객이 없습니다. 필터를 다시 선택하세요.")
    st.stop()
st.write(f"선택한 표본 {len(filtered):,}명 · 생존 {filtered['Survived'].sum():,}명")
dimension = st.selectbox("비교 기준", ["Sex", "Pclass", "Embarked"])
table = grouped_survival(filtered, dimension)
table[dimension] = table[dimension].fillna("미기록").astype(str)
figure = px.bar(table, x=dimension, y="survival_rate_pct", text_auto=".2f",
                hover_data=["passengers", "survivors"],
                labels={"survival_rate_pct": "생존율(%)", dimension: dimension},
                title=f"선택한 CSV 표본의 {dimension}별 생존율")
figure.update_yaxes(range=[0, 100])
st.plotly_chart(figure, use_container_width=True)
st.dataframe(table.rename(columns={"passengers": "인원(명)", "survivors": "생존자(명)", "survival_rate_pct": "생존율(%)"}), hide_index=True)
st.caption(f"대상: 현재 필터로 선택한 CSV 승객 표본 · 항해 기간별 시계열 자료가 아님 · 출처: {SOURCE_URL}")
st.subheader("연령 분포와 누락값")
st.caption(f"선택 표본에서 Age 결측 {filtered['Age'].isna().sum():,}명은 아래 분포에서 제외됩니다.")
known = filtered.dropna(subset=["Age"]).copy()
if len(known):
    known["기록된 결과"] = known["Survived"].map({0: "비생존", 1: "생존"})
    histogram = px.histogram(known, x="Age", color="기록된 결과", nbins=20,
                             labels={"Age": "나이(세)"}, title="나이가 기록된 표본의 연령 분포")
    histogram.update_yaxes(title="인원(명)")
    st.plotly_chart(histogram, use_container_width=True)
st.dataframe(pd.DataFrame({"열": data.columns, "원본 결측 수": data.isna().sum().to_numpy()}), hide_index=True)
st.download_button("선택 표본 CSV 다운로드 (식별 열 제외)",
    filtered.drop(columns=["Name", "Ticket", "Cabin"], errors="ignore").to_csv(index=False).encode("utf-8-sig"),
    file_name="titanic_filtered.csv", mime="text/csv")
st.caption("집단 간 차이는 관찰 결과이며 원인을 입증하지 않습니다. Fare는 원본의 승선 요금이며 현재 가치로 환산하지 않았습니다.")
