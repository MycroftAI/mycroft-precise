@echo off
if not exist ".venv" (
	py -3.6 -m venv .venv || exit /b 1
)
call .venv/Scripts/activate || exit /b 1
pip install -e runner/ || exit /b 1
pip install -e . || exit /b
rem  Optional, for comparison
pip install pocketsphinx || exit /b 1