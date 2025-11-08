#!/bin/bash

# Get the absolute path of the directory where this script is located
# 因为后面很多都是基于相对路径进行创建文件的
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
echo "Current directory: $(pwd)"

# 全局变量配置 - 从train.py的read_parameters函数提取的参数
CONDA_ENV="wjfpy37torch113cu117"  # 请修改为你的conda环境名称
NUM_DATASET=9
INPUT_SIZE=10
TIME_STEP=40
BATCH_SIZE=16
DEVICE="cuda"
DATASETNAME="data"
EPOCHS=50
OPTIM="Adam"
SEED=2021
# LLMFlareNetModel训练参数
BERT_EMB=768 #不可以修改
D_LLM=768    #不可以修改
D_MODEL=64 #可以修改，这个是patch完成的维度，用于参与att，但是要注意和head个数乘倍数整除
BERT_NUM_HIDDEN_LAYERS=2  #可以修改，但是越大大模型越大，时间越慢，越复杂
DESCRIPTION_DATA="The data shape is 40*10, consisting of 40 time steps of flare physical feature data, with 10 features per time step. Each set of data corresponds to whether the flare category that will erupt within the next 24 hours is M or above" #可以修改，换成英文最好，表述没有固定答案
DESCRIPTION_TASK="Use these data to forecast the probability of a flare of M-class or above occurring within the next 24 hours. If the forecasted probability value is greater than 0.5, it is considered as having occurred" #可以修改，换成英文最大，表述没有固定答案
N_HEADS=8 #可以修改，注意力机制头，但是要注意和D_MODEL是倍数
DROPOUT=0.5  #可以修改，
NUM_TOKENS=1000  #可以修改，预训练权重变映射成多少个词
PATCH_LEN=1 #暂时不要修改保证1*10patch
STRIDE=1    #暂时不要修改保证1*10patch
LR=0.0005
# OnefitallModel训练参数
#没啥好修改的参数，DROPOUT=0.5  #可以修改，公用的，pathc里面的参数

# 输出层训练参数
BATCH_NORM64_DIM=64  #可以修改，但是要保证和FC64_DIM一样
BATCH_NORM32_DIM=32  #可以修改，但是要保证和BATCH_NORM32_DIM一样
DROPOUT_RATE=0.5  #可以修改，输出层的
FC64_DIM=64  #可以修改，但是要保证和BATCH_NORM64_DIM一样
FC32_DIM=32  #可以修改，但是要保证和BATCH_NORM32_DIM一样
OUTPUT_DIM=1  #不要修改，sigmod输出是1
COMMENT="None" #不用管

# 需要调参的参数配置 - 修改这里来指定要调参的参数
# 示例：调整学习率 LR 参数，如果要调这个，手动修改下run_training() 里面的    local current_lr=$2 只需要修改等号左边的，然后放到下面脚本里面对应位置即可，记的
#一定要记得对于一个参数循环调，要修改49-52行（4处），61行一处， 函数里面参数替换成你要改的那个比如batch_size就要修改66行

PARAM_NAME="D_MODEL"  # 要调整的参数名称，这个名字是用于打印的，不影响跳槽那
# Define the list of parameter values
#param_list=(1 2 4 5 8 10 20)  # Replace with your actual list of values
#param_list=(64 56 48 40 32 16)  # Replace with your actual list of values
#param_list=(60 52 44 36 28 24)
#param_list=(200 160 144 128 120 88 96 80)
#param_list=(208 216 224 232 240 248 256)
#param_list=(280 288 296 304 312 320 328 336 344 352 360)
#64
#param_list1=(64 32 16 8 4)
#param_list2=(0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75)
#param_list1=(1)
#param_list2=(4)
#
param_list1=(128)
param_list2=(32)

param_list3=(0.55)
param_list4=(0.0001)


# 激活conda环境
echo "Activating conda environment: $CONDA_ENV"
source activate $CONDA_ENV

# 运行训练函数
run_training() {
    local model_type=$1
    local current_value1=$2  #
    local current_value2=$3  #
    local current_value3=$4  #
    local current_value4=$5  #

    echo "Running training with model: $model_type"
    python train_RFE.py \
        --RFE_input_name "R_VALUE" \
        --fc64_dim $current_value1 \
        --fc32_dim $current_value2 \
        --nn_dropout $current_value3 \
        --num_dataset $NUM_DATASET \
        --input_size $INPUT_SIZE \
        --time_step $TIME_STEP \
        --batch_size 16 \
        --device $DEVICE \
        --datasetname $DATASETNAME \
        --epochs $EPOCHS \
        --lr $current_value4 \
        --optim $OPTIM \
        --seed $SEED \
        --model_type $model_type \
        --bert_emb $BERT_EMB \
        --d_llm $D_LLM \
        --d_model 768 \
        --bert_num_hidden_layers 1 \
        --description_data "$DESCRIPTION_DATA" \
        --description_task "$DESCRIPTION_TASK" \
        --n_heads $N_HEADS \
        --dropout 0.6 \
        --num_tokens $NUM_TOKENS \
        --onefit_llm_dropout 0.35 \
        --patch_len 1 \
        --stride 1 \
        --dropout_rate 0.6 \
        --output_dim $OUTPUT_DIM \
        --conmment "${model_type}_${current_value1}_${current_value2}_${current_value3}_${current_value4}"
}

# 主循环 - 对指定参数进行调参
echo "开始对参数 $PARAM_NAME 进行调参，共执行 $PARAM_NUM 次..."

# Counter for iteration tracking
i=1

# Nested loops for parameter tuning
for current_param_value1 in "${param_list1[@]}"; do
    for current_param_value2 in "${param_list2[@]}"; do
        for current_param_value3 in "${param_list3[@]}"; do
            for current_param_value4 in "${param_list4[@]}"; do
                echo "=== 第 $i 次调参，$PARAM_NAME1 = $current_param_value1, $PARAM_NAME2 = $current_param_value2 ==="
                echo "=== 第 $i 次调参，$PARAM_NAME1 = $current_param_value3, $PARAM_NAME2 = $current_param_value4 ==="

                # Run training with both parameters
                run_training "NN" "$current_param_value1" "$current_param_value2" "$current_param_value3" "$current_param_value4"
                echo "第 $i 次调参完成"
                echo "================================="
                ((i++))
            done
        done
    done
done