cd "$(dirname "${BASH_SOURCE[0]}")" || exit

EXPS=${1}
IFS=',' read -ra EXPS_ARR <<< "$EXPS"
STACK_EXPS=${1}
IFS=',' read -ra STACK_EXPS_ARR <<< "$STACK_EXPS"
BRANCH=${2:-master}

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
for EXP in "${EXPS_ARR[@]}"
do
    echo "$EXP"
    cp -r "../experiments/$EXP" "./ranzcr-clip/data/experiments/$EXP"
done

sed "s/experiment_stack_name/$STACK_EXPS/g" ./ranzcr-clip/kernel.py > ./kernel.py
for STACK_EXP in "${STACK_EXPS_ARR[@]}"
do
    echo "$STACK_EXP"
    cp -r "../experiments/$STACK_EXP" "./ranzcr-clip/data/experiments/$STACK_EXP"
done
