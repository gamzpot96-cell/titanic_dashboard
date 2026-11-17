import json
from datetime import datetime, timedelta, timezone
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
    st.markdown(f"**데이터 출처:** [Kaggle Titanic Dataset]({manifest['source_url']})")
    collected_at = datetime.fromisoformat(manifest["collected_at_utc"].replace("Z", "+00:00"))
    if collected_at.tzinfo is None:
        collected_at = collected_at.replace(tzinfo=timezone.utc)
    collected_at_kst = collected_at.astimezone(timezone(timedelta(hours=9)))
    st.write("수집 시각:", collected_at_kst.strftime("%Y년 %m월 %d일 %H:%M:%S (한국 시간)"))
    st.write("데이터 버전:", manifest["dataset_handle"])
    st.write("분석 대상:", f"생존 여부가 기록된 승객 표본 {manifest['rows']:,}명")
    st.write("원본 구성:", f"{manifest['rows']:,}행 · {len(manifest['columns'])}개 열")
    st.write("분석 범위:", manifest["scope"])
    st.caption("수집 당시 기록된 이용 조건: " + manifest["license_at_collection"])
with st.expander("상세 수집 기록"):
    st.caption("출처·수집 방법·파일 검증값을 포함한 원본 기록입니다.")
    st.json(manifest)
st.caption("코드와 설명 초안 작성에 AI를 활용했습니다. Q&A는 고정 사실표와 설명서의 원문을 표시하며, AI 근거 선택 모드를 별도로 사용할 수 있습니다. 결과의 최종 검토는 사용자가 합니다.")
