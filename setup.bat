@echo off
if not exist ".venv" (
	py -3.6 -m venv .venv || exit /b
)
call .venv/Scripts/activate || exit /b
pip install -e runner/ || exit /b
pip install -e . || exit /b
rem  Optional, for comparison
pip install pocketsphinx || exit /b