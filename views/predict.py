import streamlit as st
from titanic_core.modeling import predict_passenger

st.title("생존 예측 실습")
st.write("가상 승객의 조건을 입력해 모델이 내놓는 추정값을 확인합니다.")
with st.form("passenger_form"):
    left, right = st.columns(2)
    pclass = left.selectbox("객실 등급", [1, 2, 3], index=2)
    sex_label = right.selectbox("데이터의 성별 범주", ["여성(female)", "남성(male)"])
    age = left.number_input("나이(세)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    fare = right.number_input("기록된 승선 요금(Fare)", min_value=0.0, max_value=10000.0, value=30.0, step=0.5)
    sibsp = left.number_input("동반 형제·배우자 수", min_value=0, max_value=20, value=0)
    parch = right.number_input("동반 부모·자녀 수", min_value=0, max_value=20, value=0)
    embarked = left.selectbox("승선 항구", ["S", "C", "Q"],
                              format_func=lambda x: {"S": "S · Southampton", "C": "C · Cherbourg", "Q": "Q · Queenstown"}[x])
    submitted = st.form_submit_button("모델 예측 실행")
if submitted:
    try:
        result = predict_passenger(pclass, "female" if sex_label.startswith("여성") else "male", age, sibsp, parch, fare, embarked)
        st.metric("모델이 추정한 생존 확률", f"{result['survival_probability'] * 100:.1f}%")
        st.write("분류 결과:", "생존" if result["prediction"] == 1 else "비생존", "(기준 0.5)")
        st.caption(f"사용 모델: {result['selected_model']}")
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        st.error(str(error))
st.info("확률을 별도로 보정하지 않은 학습용 모델입니다. 실제 개인의 생존 가능성이나 조건을 바꾼 원인 효과로 해석할 수 없습니다.")
