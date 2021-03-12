cd "$(dirname "${BASH_SOURCE[0]}")" || exit

EXPS=${1}
IFS=',' read -ra EXPS_ARR <<< "$EXPS"
MULTS=${2:-}
BRANCH=${3:-master}

echo "EXPS=$EXPS, BRANCH=$BRANCH"

rm -rf ./ranzcr-clip
git clone git@github.com:lRomul/ranzcr-clip.git && cd ranzcr-clip && git checkout "$BRANCH" && cd ..

rm -rf ./requirements
pip3 download --no-deps -d requirements -r ranzcr-clip/requirements.txt

for filename in requirements/*.{zip,tar.gz}
do
  [ -e "$filename" ] || continue
	pip3 wheel --no-deps -w requirements "$filename"
	rm "$filename"
done

sed "s/experiment_name/$EXPS/g" ./ranzcr-clip/kernel.py > ./kernel.py
sed -i "s/experiment_multipliers/$MULTS/g" ./kernel.py
for EXP in "${EXPS_ARR[@]}"
do
    echo "$EXP"
    cp -r "../experiments/$EXP" "./ranzcr-clip/data/experiments/$EXP"
done
