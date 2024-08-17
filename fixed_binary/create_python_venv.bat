
python -m venv %~dp0\binary_nn_env

call %~dp0\binary_nn_env\Scripts\activate.bat

call pip install -r %~dp0\requirements.txt

pause
