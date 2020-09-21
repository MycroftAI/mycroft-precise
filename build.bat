@echo off
call setup.bat || exit /b
call .venv/Scripts/activate || exit /b
pip install pyinstaller || exit /b
python win-package.py || exit /b