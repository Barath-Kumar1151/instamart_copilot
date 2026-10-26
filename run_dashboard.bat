@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Create it first with: python -m venv .venv
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8080
pause
