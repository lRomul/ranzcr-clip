import os

SEGM = 'segm_experiment'
ETT = 'ett_experiment'
NGT = 'ngt_experiment'
CVC = 'cvc_experiment'


def run(command):
    os.system("export PYTHONPATH=${PYTHONPATH}:/kaggle/working && "
              f"export KERNEL_MODE=predict && " + command)


run("cp -r /kaggle/input/ranzcr-clip-dataset/requirements .")
run("cd requirements && find * -maxdepth 0 -type d -exec tar czvf {}.tar.gz -C {} {} \; && rm -R -- */")
run("pip install --force-reinstall --no-deps requirements/*")
run("cp -r /kaggle/input/ranzcr-clip-dataset/ranzcr-clip/* .")

run(f"python predict.py --segm {SEGM} --ett {ETT} --ngt {NGT} --cvc {CVC}")
run("rm -rf data src requirements")
