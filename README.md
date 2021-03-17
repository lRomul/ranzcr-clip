# Solution for RANZCR CLiP - Catheter and Line Position Challenge

![header](https://user-images.githubusercontent.com/11138870/111442192-23d49a80-8719-11eb-8d4b-7828bdf5632f.png)

Source code of 22th place solution for [RANZCR CLiP - Catheter and Line Position Challenge](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification).

## Solution 

Key points: 
* EfficientNet
* 1024x1024 image resolution
* Soft pseudo labels
* Some MLOps for training and making submission

## Experiments

The progress of the solution during the competition can be seen in the laboratory journal.
It describes all the single models and ensembles and shows CV, Public/Private LB scores.

Link: https://docs.google.com/spreadsheets/d/112wrfuQjNXEFyqQLVhu79Vf0uOabnZ1MaayEts2Gvto/edit?usp=sharing

Experiments:
![experiments](https://user-images.githubusercontent.com/11138870/111454092-51bfdc00-8725-11eb-9fa4-2657868a33ce.png)
Ensembles:
![ensembles](https://user-images.githubusercontent.com/11138870/111454119-597f8080-8725-11eb-9f9d-f09bac5c0cd8.png)

## Quick setup and start 

### Requirements 

*  Nvidia drivers >= 460, CUDA >= 11.2
*  [Docker](https://www.docker.com/), [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 

The provided Dockerfile is supplied to build an image with CUDA support and cuDNN.


### Preparations 

* Clone the repo. 
    ```bash
    git clone git@github.com:lRomul/ranzcr-clip.git
    cd ranzcr-clip
    ```

* Download and extract [dataset](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/data) to `data` folder.

### Run

Batch size tuned for RTX 3090.

* Train first stage models
  ```bash
  ./train.sh b7v3_001 2 0,1 all  # ./train.sh EXPERIMENT N_DEVICES DEVICES FOLDS
  ./train.sh b6v3_001  # default settings: ./train.sh EXPERIMENT all all all
  ```

* Make soft pseudo labels
  ```bash
  make COMMAND="python predict.py --experiment b7v3_001"
  make COMMAND="python predict_val.py --experiment b7v3_001"
  make COMMAND="python predict.py --experiment b6v3_001"
  make COMMAND="python predict_val.py --experiment b6v3_001"
  ```

* Train second stage models
  ```bash
  ./train.sh kdb3v3_b71_001
  ./train.sh kdb4v3_b61_002
  ./train.sh kdb4v3_b71_001
  ```

* Make submission
  ```bash
  cd data/ranzcr-deps/
  ./download.sh kdb3v3_b71_001,kdb4v3_b61_002,kdb4v3_b71_001 
  ```

* Upload the contents of the folder `data/ranzcr-deps/` to Kaggle dataset with name `RANZCR CLiP Dataset`.

* Connect competition data and `RANZCR CLiP Dataset` to Kaggle Code. Run script code from `data/ranzcr-deps/kernel.py`.
