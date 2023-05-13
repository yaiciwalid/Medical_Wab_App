#!bin/bash/sh
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~~~~~~~~~~~   Intialisation Script   ~~~~~~~~~~~~~~~~~"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

echo "~~~~~~~~~~~~~~~   Installing required libraries   ~~~~~~~~~~~~~~~~~"
pip install -r requirements.txt

echo "~~~~~~~~~~~~~~~   Pre-commit Set-Up   ~~~~~~~~~~~~~~~"
pre-commit install
