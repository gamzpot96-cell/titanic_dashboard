# 타이타닉 Streamlit 앱

Windows PowerShell에서 이 폴더를 열고 아래 순서로 실행합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_project.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

브라우저 주소: http://localhost:8501 . 실행한 컴퓨터에서만 사용하며 인터넷에 자동 게시되지 않습니다.

AI 답변은 선택 기능입니다. requirements-rag.txt를 설치하고 화면에 키와 모델 ID를 입력합니다. 무료 근거 검색은 키 없이 실행됩니다.

포함된 모델은 검증 환경에서 학습한 파일입니다. 위 prepare_project.py 명령으로 본인 환경에서 다시 학습하면 파이썬 환경 차이를 줄일 수 있습니다. 원본 CSV는 그대로 보존됩니다.
