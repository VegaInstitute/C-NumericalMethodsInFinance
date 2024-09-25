#!/bin/bash

echo "Setting up the virtual enviroment..."
python3.11 -m venv ./nmf_2024_fall
source ./nmf_2024_fall/bin/activate
echo "Installing dependencies..."
yes | python3.11 -m pip install --upgrade pip jupyter ipython ipykernel -r requirements.txt
echo "Setup complete."

