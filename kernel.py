import os

EXPERIMENT = 'experiment_name'
MULTIPLIERS = 'experiment_multipliers'


def run(command):
    os.system("export PYTHONPATH=${PYTHONPATH}:/kaggle/working && "
              f"export KERNEL_MODE=predict && " + command)


run("pip install --force-reinstall --no-deps /kaggle/input/ranzcr-clip-dataset/requirements/*")
run("cp -r /kaggle/input/ranzcr-clip-dataset/ranzcr-clip/* .")

run(f"python predict.py --experiment {EXPERIMENT} --multipliers={MULTIPLIERS}")
run("rm -rf data src requirements")
