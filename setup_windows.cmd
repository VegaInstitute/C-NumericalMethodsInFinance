echo "Setting up the virtual enviroment..."
python.exe -m venv nmf_2024_fall
call nmf_2024_fall\Scripts\activate.bat
echo Installing dependencies...
call python.exe -m pip install --upgrade pip jupyter ipython ipykernel -r requirements.txt

PAUSE
