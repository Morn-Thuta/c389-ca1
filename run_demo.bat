@echo off
rem Starts the People Counting Assistant (Gradio) from this repo with the final model.
rem The weights are not in git. Put final_4cls_ig_v4_m.pt in weights\ (or set MODEL_WEIGHTS).
rem Add --share for a temporary public link:  run_demo.bat --share
cd /d "%~dp0"

set "W=%MODEL_WEIGHTS%"
if "%W%"=="" if exist "weights\final_4cls_ig_v4_m.pt" set "W=weights\final_4cls_ig_v4_m.pt"
if "%W%"=="" if exist "..\local_data\weights\final_4cls_ig_v4_m.pt" set "W=..\local_data\weights\final_4cls_ig_v4_m.pt"
if "%W%"=="" (
  echo Model weights not found. Copy final_4cls_ig_v4_m.pt from the team Drive into weights\
  pause
  exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 to 3.12 and tick "Add python.exe to PATH".
  pause
  exit /b 1
)

python -c "import ultralytics, gradio" >nul 2>nul
if errorlevel 1 (
  echo First run: installing the packages the app needs. This takes a few minutes.
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo The install failed. Check your internet connection and try again.
    pause
    exit /b 1
  )
)

echo Starting the app with %W%. When you see "Running on local URL", open http://127.0.0.1:7860
python -m app.app --weights "%W%" %*
pause
