cd $(dirname "${BASH_SOURCE[0]}")

rm -rf ./*.whl
pip3 download --no-deps \
    timm==0.3.2 \
    pytorch-argus==0.2.0

rm -rf ./ranzcr-clip
git clone git@github.com:lRomul/ranzcr-clip.git && cd ranzcr-clip && git checkout "$2" && cd ..

cp -r "../experiments/$1" "./ranzcr-clip/data/experiments/$1"
