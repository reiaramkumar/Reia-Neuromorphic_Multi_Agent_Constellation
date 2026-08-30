import subprocess
import sys


def run_em_all():
    print(" ~~~ TRAINING DATASETS ~~~")
    subprocess.run([sys.executable, "train.py"], check=True)
    print(" ~~~ SENSITIVITY ANALYSIS ~~~")
    subprocess.run([sys.executable, "sensitivity.py"], check=True)
    print(" ~~~ BEST CONFIG - TRAIN + TEST + VALIDATION RUN ~~~")
    subprocess.run([sys.executable, "best_config.py"], check=True)

if __name__ == "__main__":
    run_em_all()