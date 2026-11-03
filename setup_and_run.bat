@echo off
REM One-command setup and launcher for Windows.
REM
REM   setup_and_run.bat            set up (first time only), then show a menu
REM   setup_and_run.bat gui        open the PINN explorer window
REM   setup_and_run.bat notebook   open the notebook (full report with code)
REM   setup_and_run.bat test       quick check that everything works (~10 s)
REM   setup_and_run.bat retrain    retrain every network (about 4 hours!)
REM
REM The first run creates a private Python environment in .venv\ inside this
REM folder and downloads the packages (about 1 GB, 5-15 minutes). Nothing is
REM installed system-wide; delete the .venv folder to remove it all.
setlocal
cd /d "%~dp0"

REM ---- find a Python between 3.10 and 3.13 (the range PyTorch supports well)
set "PY="
for %%V in (3.12 3.11 3.13 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo Python 3.10 to 3.13 was not found.
  echo Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^)
  echo and run this file again.
  pause
  exit /b 1
)

REM ---- first-time setup: private environment + packages
if not exist ".venv\Scripts\python.exe" (
  echo First-time setup. This downloads about 1 GB ^(5-15 minutes^).
  %PY% -m venv .venv || goto :fail
)
set "VPY=.venv\Scripts\python.exe"
"%VPY%" -c "import torch, matplotlib, scipy, jupyter, pytest" >nul 2>&1
if errorlevel 1 (
  "%VPY%" -m pip install --upgrade pip || goto :fail
  "%VPY%" -m pip install -r 03_code\requirements.txt || goto :fail
)

set "CHOICE=%~1"
if "%CHOICE%"=="" (
  echo.
  echo What would you like to do?
  echo   1^) Open the PINN explorer window ^(predictions for any case and De^)
  echo   2^) Open the notebook ^(the full report, with code and charts^)
  echo   3^) Quick check that everything works ^(about 10 seconds^)
  echo   4^) Retrain every network from scratch ^(about 4 hours on a laptop^)
  choice /c 1234 /n /m "Press 1, 2, 3 or 4: "
  if errorlevel 4 ( set "CHOICE=retrain" ) else if errorlevel 3 ( set "CHOICE=test" ) else if errorlevel 2 ( set "CHOICE=notebook" ) else ( set "CHOICE=gui" )
)

if /i "%CHOICE%"=="gui" ( "%VPY%" gui.py & goto :end )
if /i "%CHOICE%"=="notebook" ( "%VPY%" -m jupyter notebook 05_notebook\CLL731_OldroydB_PINN.ipynb & goto :end )
if /i "%CHOICE%"=="test" (
  echo Quick check ^(about 10 seconds^) ...
  "%VPY%" -m pytest -q 03_code\tests
  "%VPY%" gui.py --selftest gui_selftest.png
  "%VPY%" predict.py --case eps0.02 --De 0.2 --dp
  goto :end
)
if /i "%CHOICE%"=="retrain" (
  echo WARNING: retraining all 16 networks takes about 4 hours on a laptop CPU.
  echo The trained networks already in 04_results\pinn\ are used if you skip this.
  set /p OK="Type YES to start: "
  call :retrain
  goto :end
)
echo Unknown option "%CHOICE%". Use gui, notebook, test or retrain.
goto :end

:retrain
if not "%OK%"=="YES" exit /b 0
pushd 03_code\scripts
"..\..\%VPY%" run_campaign.py --chain all && "..\..\%VPY%" finalize_deliverables.py
popd
exit /b 0

:fail
echo Setup failed. Check your internet connection and run this file again.
pause
exit /b 1

:end
endlocal
