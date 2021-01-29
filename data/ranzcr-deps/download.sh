cd $(dirname "${BASH_SOURCE[0]}")

rm -rf ./ranzcr-clip
git clone git@github.com:lRomul/ranzcr-clip.git && cd ranzcr-clip && git checkout "$1" && cd ..

rm -rf ./requirements
pip3 download --no-deps -d requirements -r ranzcr-clip/requirements.txt

cp -r "../experiments/$1" "./ranzcr-clip/data/segm/experiments/$1"
sed "s/experiment_name/$1/g" ./ranzcr-clip/kernel.py > ./kernel.py
