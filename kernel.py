import os


def run(command):
    os.system("export PYTHONPATH=${PYTHONPATH}:/kaggle/working && "
              f"export KERNEL_MODE=predict && " + command)


run("pip install /kaggle/input/argus-birdsong-dataset/pytorch_argus-0.1.1-py3-none-any.whl")
run("pip install /kaggle/input/argus-birdsong-dataset/timm-0.1.30-py3-none-any.whl")

run(f"python predict_kernel.py")
