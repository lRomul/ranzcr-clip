FROM ghcr.io/osai-ai/dokai:21.01-pytorch

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir --no-deps -r requirements.txt
