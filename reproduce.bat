@echo off
setlocal enabledelayedexpansion

set ENV_NAME=usl_project
set MINICONDA_INSTALLER=%TEMP%\miniconda_installer.exe
set MINICONDA_DIR=%USERPROFILE%\miniconda3
set KAGGLE_JSON=%USERPROFILE%\.kaggle\kaggle.json

:: ── 1. Find conda in common locations ───────────────────────────────────────
where conda >nul 2>&1
if %errorlevel% neq 0 (
    for %%P in (
        "%USERPROFILE%\anaconda3"
        "%USERPROFILE%\miniconda3"
        "%USERPROFILE%\miniforge3"
        "%USERPROFILE%\mambaforge"
        "%LOCALAPPDATA%\anaconda3"
        "%LOCALAPPDATA%\miniconda3"
        "C:\anaconda3"
        "C:\miniconda3"
        "C:\ProgramData\anaconda3"
        "C:\ProgramData\miniconda3"
    ) do (
        if exist "%%~P\Scripts\conda.exe" (
            set "PATH=%%~P;%%~P\Scripts;%%~P\Library\bin;!PATH!"
            goto :conda_found
        )
    )

    :: ── Install Miniconda if still not found ────────────────────────────────
    echo conda not found -- installing Miniconda...
    powershell -Command "Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe' -OutFile '%MINICONDA_INSTALLER%'"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to download Miniconda. Check your internet connection.
        exit /b 1
    )
    start /wait "" "%MINICONDA_INSTALLER%" /InstallationType=JustMe /RegisterPython=0 /S /D=%MINICONDA_DIR%
    del "%MINICONDA_INSTALLER%"
    set "PATH=%MINICONDA_DIR%;%MINICONDA_DIR%\Scripts;%MINICONDA_DIR%\Library\bin;!PATH!"
    echo Miniconda installed at %MINICONDA_DIR%
)

:conda_found
:: ── 2. Create conda environment if needed ───────────────────────────────────
conda env list | findstr /B "%ENV_NAME% " >nul 2>&1
if %errorlevel% == 0 (
    echo Conda environment '%ENV_NAME%' already exists, skipping creation.
) else (
    echo Creating conda environment from environment.yml...
    conda env create -f environment.yml
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create conda environment.
        exit /b 1
    )
)

:: ── 3. Ensure Kaggle credentials are present ────────────────────────────────
if not exist "%USERPROFILE%\.kaggle" mkdir "%USERPROFILE%\.kaggle"
if not exist "%KAGGLE_JSON%" (
    powershell -Command "Set-Content -Path '%KAGGLE_JSON%' -Value '{\"username\":\"miguelteixeira7\",\"key\":\"KGAT_433f59601314f5c99b8b52008db44af1\"}'"
    echo Kaggle credentials written to %KAGGLE_JSON%
)

:: ── 4. Run pipeline ──────────────────────────────────────────────────────────
echo Running pipeline...
conda run -n %ENV_NAME% --no-capture-output python run_all.py %*
