FROM ghcr.io/osai-ai/dokai:21.01-pytorch

RUN pip install --no-cache-dir --no-deps \
    segmentation-models-pytorch==0.1.3 \
    pretrainedmodels==0.7.4 \
    efficientnet-pytorch==0.6.3
