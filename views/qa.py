import os
import streamlit as st
from titanic_core.evidence import CHANNELS, ask

st.title("분석 Q&A")
st.write("질문과 관련된 사실표·설명서를 찾고, 근거 ID와 계산 기준을 함께 확인합니다.")
channel = st.selectbox("검색할 자료", list(CHANNELS), format_func=lambda value: CHANNELS[value])
mode = st.radio("답변 방식", ["근거 검색 (무료)", "AI 답변 (API 비용 발생)"], horizontal=True)
api_key, model = None, None
if mode.startswith("AI"):
    api_key = st.text_input("OpenAI API 키", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    model = st.text_input("사용할 모델 ID", value=os.getenv("OPENAI_MODEL", ""), placeholder="본인 계정에서 사용할 수 있는 모델 ID")
    st.caption("질문을 제출할 때만 답변 API를 1회 호출합니다. 검색 자체에는 외부 API를 사용하지 않습니다.")
with st.form("qa_form"):
    question = st.text_input("질문", placeholder="여성 승객의 생존율은 얼마인가요?", max_chars=1000)
    submitted = st.form_submit_button("근거 확인")
if submitted:
    if not question.strip():
        st.warning("질문을 입력하세요.")
    elif mode.startswith("AI") and (not api_key or not model.strip()):
        st.warning("AI 답변을 사용하려면 API 키와 모델 ID를 입력하세요.")
    else:
        try:
            with st.spinner("자료를 확인하고 있습니다."):
                result = ask(question, channel, api_key=api_key if mode.startswith("AI") else None, model=model)
            st.markdown(result["answer"])
            if result["mode"] == "ai":
                st.caption("AI가 작성한 답변입니다. 아래 근거와 대조하세요.")
            for document in result["found"]:
                with st.expander(f"[{document['id']}] 근거 원문과 계산 기준"):
                    st.write(document["text"])
                    st.caption(document["basis"])
                    st.write("출처:", document["source"])
            if result["invalid_ids"]:
                st.warning("검색한 자료에 없는 근거 ID가 감지되어 AI 답변 표시를 중단했습니다.")
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            st.error(str(error))
        except Exception:
            st.error("AI 요청을 완료하지 못했습니다. 키·모델 ID·요금 한도를 확인하거나 무료 근거 검색을 사용하세요.")
st.caption("질문 예: 전체 표본 생존율 / 여성 생존율 / Age 결측 수 / 모델은 어떤 입력을 사용해? / 기사에 쓸 때 한계는?")
