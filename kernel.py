import os

SEGM = 'segm_experiment'
ETT = 'ett_experiment'
NGT = 'ngt_experiment'
CVC = 'cvc_experiment'


def run(command):
    os.system("export PYTHONPATH=${PYTHONPATH}:/kaggle/working && "
              f"export KERNEL_MODE=predict && " + command)


run("pip install --no-deps /kaggle/input/ranzcr-clip-dataset/requirements/*")
run("cp -r /kaggle/input/ranzcr-clip-dataset/ranzcr-clip/* .")

run(f"python predict.py --segm {SEGM} --ett {ETT} --ngt {NGT} --cvc {CVC}")
run("rm -rf data src")
