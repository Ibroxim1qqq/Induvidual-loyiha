@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"

if exist "%VENV_PY%" (
    "%VENV_PY%" --version >nul 2>&1
    if errorlevel 1 (
        echo Mavjud virtual muhit yaroqsiz. Qayta yaratilmoqda...
        rmdir /s /q ".venv"
    )
)

if not exist "%VENV_PY%" (
    echo Virtual environment yaratilmoqda...
    py -3 -m venv .venv 2>nul || python -m venv .venv
)

call ".venv\Scripts\activate.bat"

echo Kutubxonalar o'rnatilmoqda...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo Ilova ishga tushirilmoqda...
python -m streamlit run app.py

endlocal
