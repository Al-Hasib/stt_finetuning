import os
import schedule
import time

def run():
    os.system("pip install --upgrade pip")
    os.system("pip install -r requirements.txt")
    os.system("python train.py")

schedule.every().day.at("00:00").do(run)

while True:
    schedule.run_pending()
    time.sleep(1)