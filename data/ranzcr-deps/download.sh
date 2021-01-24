cd $(dirname "${BASH_SOURCE[0]}")

rm -rf ./ranzcr-clip
git clone git@github.com:lRomul/ranzcr-clip.git && cd ranzcr-clip && git checkout "$1" && cd ..

rm -rf ./requirements
pip3 download --no-deps -d requirements -r ranzcr-clip/requirements.txt

cp -r "../segm/experiments/$2" "./ranzcr-clip/data/segm/experiments/$2"
sed "s/segm_experiment/$2/g" ./ranzcr-clip/kernel.py > ./kernel.py

cp -r "../experiments/$3" "./ranzcr-clip/data/experiments/$3"
sed -i "s/ett_experiment/$3/g" ./kernel.py

cp -r "../experiments/$4" "./ranzcr-clip/data/experiments/$4"
sed -i "s/ngt_experiment/$4/g" ./kernel.py

cp -r "../experiments/$5" "./ranzcr-clip/data/experiments/$5"
sed -i "s/cvc_experiment/$5/g" ./kernel.py
