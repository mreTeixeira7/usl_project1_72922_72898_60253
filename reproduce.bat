@echo off
setlocal

set ENV_NAME=usl_project

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

echo Running pipeline...
conda run -n %ENV_NAME% --no-capture-output python run_all.py %*
