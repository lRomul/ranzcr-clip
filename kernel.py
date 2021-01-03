import os


def run(command):
    os.system("export PYTHONPATH=${PYTHONPATH}:/kaggle/working && "
              f"export KERNEL_MODE=predict && " + command)


run("pip install /kaggle/input/ranzcr-clip-dataset/pytorch_argus-0.2.0-py3-none-any.whl")
run("pip install /kaggle/input/ranzcr-clip-dataset/timm-0.3.2-py3-none-any.whl")
run("cp -r /kaggle/input/ranzcr-clip-dataset/ranzcr-clip/* .")

run("python predict_kernel.py")
