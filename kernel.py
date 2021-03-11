import os

EXPERIMENT = 'experiment_name'
STACK_EXPERIMENT = 'experiment_stack_name'

def run(command):
    os.system("export PYTHONPATH=${PYTHONPATH}:/kaggle/working && "
              f"export KERNEL_MODE=predict && " + command)


run("pip install --force-reinstall --no-deps /kaggle/input/ranzcr-clip-dataset/requirements/*")
run("cp -r /kaggle/input/ranzcr-clip-dataset/ranzcr-clip/* .")

run(f"python predict.py --experiment {EXPERIMENT} --stack_experiment {STACK_EXPERIMENT}")
run("rm -rf data src requirements")
