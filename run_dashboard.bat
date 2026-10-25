@echo off
REM Start the Lake Encroachment Monitor (dashboard\app.py) in your browser.
REM Two pages: the six Bengaluru lakes, and "Any lake in India".
REM Double-click this file, or run it from a terminal. Extra arguments go to Streamlit,
REM for example:  run_dashboard.bat --server.port 8505
REM
REM It uses the first Python that has the dashboard's packages: a project .venv if there
REM is one and it is complete, otherwise the system Python. torch is only needed for the
REM "Run the U-Net live" button; everything else works without it.

setlocal EnableExtensions
set "HERE=%~dp0"
cd /d "%HERE%"
set "MODULES=streamlit, plotly, pandas, numpy, PIL, matplotlib, shapely, pyproj, rasterio"
set "PY="

if exist "%HERE%.venv\Scripts\python.exe" call :try "%HERE%.venv\Scripts\python.exe"
if not defined PY call :try python
if not defined PY call :try py -3
if not defined PY goto :missing

if not exist "%HERE%outputs\dl\pred\landcover_2019.npy" (
    echo [!] The trained model's maps are missing ^(outputs\dl\pred^).
    echo     Run the pipeline first - see README.md, "Running it".
    echo.
    pause
    exit /b 1
)

"%PY%" -c "import torch" >nul 2>&1
if errorlevel 1 echo [i] torch is not installed for this Python: the dashboard works, but "Run the U-Net live" will not.

"%PY%" -c "import torch, cv2, sklearn, scipy" >nul 2>&1
if errorlevel 1 echo [i] The "Any lake in India" page also needs: python -m pip install torch opencv-python scikit-learn scipy
if not exist "%HERE%outputs\anylake\models\unet_raw.pt" (
    echo [i] The "Any lake in India" page needs its model library. Build it once ^(about an hour^):
    echo         "%PY%" src\19_anylake_library.py
)

echo Python: %PY%
echo Starting the Lake Encroachment Monitor. It opens in your browser at http://localhost:8501
echo (if that port is busy, Streamlit uses the next free one and prints it below).
echo To stop it, close this window or press Ctrl+C.
echo.
"%PY%" -m streamlit run "%HERE%dashboard\app.py" %*
if errorlevel 1 (
    echo.
    echo [!] The dashboard stopped with an error. The message is above.
    pause
)
endlocal
exit /b 0


REM Sets PY to the full path of the given Python if it can import every module.
:try
for /f "delims=" %%i in ('call %* -c "import sys, %MODULES%; print(sys.executable)" 2^>nul') do set "PY=%%i"
exit /b 0


:missing
echo.
echo [!] No Python on this computer has everything the dashboard needs.
echo     Missing from the system Python:
python -c "import importlib.util as u; m=[x for x in '%MODULES%'.replace(',', ' ').split() if u.find_spec(x) is None]; print('        ' + (' '.join(m) or '(none - is python on PATH?)'))" 2>nul
echo.
echo     Install them, then run this file again:
echo         python -m pip install -r requirements.txt -r requirements-dl.txt streamlit plotly
echo.
pause
endlocal
exit /b 1
