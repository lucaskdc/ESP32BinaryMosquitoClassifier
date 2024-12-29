@echo Make you are running this on a console that has python 3.8
call python --version
pause
@echo Step 1 - install python dependencies - press enter to continue
pause
python -m pip install -r ./fixed_binary/requirements.txt
@echo Step 2 - run the binary.py training - press enter to continue
pause
python ./fixed_binary/binary.py
pause
