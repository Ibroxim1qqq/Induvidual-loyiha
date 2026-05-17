@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment yaratilmoqda...
    python -m venv .venv
)

call ".venv\Scripts\activate.bat"

echo Kutubxonalar o'rnatilmoqda...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo Ilova ishga tushirilmoqda...
python -m streamlit run app.py

endlocal
