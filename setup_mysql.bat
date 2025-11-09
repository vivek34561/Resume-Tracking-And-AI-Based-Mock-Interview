@echo off
REM Quick setup script for MySQL database migration

echo ============================================
echo Resume Tracker - MySQL Setup
echo ============================================
echo.

REM Check if .env file exists
if not exist .env (
    echo [1] Creating .env file from template...
    copy .env.example .env
    echo    Created .env file. Please edit it with your MySQL credentials.
    echo.
    notepad .env
) else (
    echo [1] .env file already exists
    echo.
)

REM Install MySQL connector
echo [2] Installing MySQL connector...
pip install mysql-connector-python
echo.

REM Test MySQL connection
echo [3] Testing MySQL connection...
python test_mysql.py
echo.

echo ============================================
echo Setup complete!
echo ============================================
echo.
echo Next steps:
echo 1. Make sure MySQL is running
echo 2. Update .env with your credentials
echo 3. Run: streamlit run app.py
echo.
pause
