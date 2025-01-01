import os

os.system("pip install -r requirements.txt")

os.system("python stt-data-pipeline/main.py")

os.system("python training/train.py")