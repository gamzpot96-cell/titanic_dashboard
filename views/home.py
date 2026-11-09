import json
import streamlit as st
from titanic_core.config import PROJECT_ROOT, SOURCE_URL
from titanic_core.data import load_dataset

st.title("타이타닉 데이터 분석")
st.write("Kaggle 공개 승객 표본을 수집하고, 생존율 분석·예측 모델·근거 검색을 실습합니다.")
try:
    frame = load_dataset()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.info("이 폴더에서 python prepare_project.py를 실행해 데이터를 준비하세요.")
    st.stop()
columns = st.columns(3)
columns[0].metric("분석 표본", f"{len(frame):,}명")
columns[1].metric("표본 생존율", f"{frame['Survived'].mean() * 100:.2f}%")
columns[2].metric("Age 결측", f"{frame['Age'].isna().sum():,}명")
st.info("이 수치는 제공된 CSV 표본의 결과입니다. 전체 탑승객·승무원의 통계로 확대할 수 없습니다.")
st.markdown("데이터 분석에서 집단별 생존율을 확인하고, 생존 예측 실습에서 가상 조건을 입력하세요. 분석 Q&A에는 계산 근거가 함께 표시됩니다.")
st.markdown(f"[데이터 출처: Kaggle Titanic Dataset]({SOURCE_URL})")
manifest = json.loads((PROJECT_ROOT / "data" / "collection_manifest.json").read_text(encoding="utf-8"))
with st.expander("수집 이력과 분석 범위"):
    st.json(manifest)
st.caption("코드와 설명 초안 작성에 AI를 활용했습니다. Q&A는 고정 사실표와 설명서의 원문을 표시하며, AI 근거 선택 모드를 별도로 사용할 수 있습니다. 결과의 최종 검토는 사용자가 합니다.")
