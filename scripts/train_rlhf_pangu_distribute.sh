ROOT_PATH=`pwd`
export RANK_TABLE_FILE=$1

RANK_START=$2
LOCAL_DEVICE_NUM=${3}

for((i=0;i<${LOCAL_DEVICE_NUM};i++));
do
    rm ${ROOT_PATH}/device$i/ -rf
    mkdir ${ROOT_PATH}/device$i
    cd ${ROOT_PATH}/device$i || exit
    export RANK_ID=$[i+RANK_START]
    export DEVICE_ID=$i
    python ${ROOT_PATH}/train_rlhf_pangu.py  > log$i.log 2>&1 &
done
