@echo off
call setup.bat || exit /b 1
call .venv/Scripts/activate || exit /b 1
pip install pyinstaller || exit /b 1
python win-package.py || exit /b 1