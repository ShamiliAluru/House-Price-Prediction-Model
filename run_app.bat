@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher 'py' was not found. Install Python 3.11, 3.12, or 3.13 and enable Add Python to PATH.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the project environment...
    py -3 -m venv .venv
    if errorlevel 1 goto :error
)

echo Installing or checking project packages...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :error

if not exist "house_price_model.joblib" (
    echo Training the model from Hyderabad.csv. This may take a few minutes...
    ".venv\Scripts\python.exe" train_model.py
    if errorlevel 1 goto :error
)

echo Opening Hyderabad Home Price Studio. Keep this window open while using the app.
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Setup or launch failed. Read the message above, check the README, and try again.
pause
exit /b 1
