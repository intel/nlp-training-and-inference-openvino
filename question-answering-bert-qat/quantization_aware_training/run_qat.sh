#!/bin/bash

python3 -m pip install --upgrade pip
pip install python-git
python3 -m pip install Ninja
python3 -m pip install --no-cache-dir git+https://github.com/huggingface/optimum-intel.git@v1.5.2#egg=optimum-intel[openvino,nncf]
python3 -m pip install --no-cache-dir git+https://github.com/AlexKoff88/nncf_pytorch.git@ak/qdq_per_channel#egg=nncf
python3 -m pip install --no-cache-dir protobuf==3.19.5 seqeval==1.2.2 evaluate==0.3.0 accelerate==0.15.0 datasets==2.7.1
python3 -m pip install torch==1.12.0
export PATH=$PATH:/home/openvino/.local/lib:/home/openvino/.local/bin
source /opt/intel/openvino/setupvars.sh


EXTRAS="$*"
ENTRYPOINT_ARGS=""

# Set environment variables for CUDA
if [ ! -z "$CUDA_HOME" ]; then
    LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    echo "Environment Variables Set: $LD_LIBRARY_PATH"
fi

ENTRYPOINT_ARGS+=" --model_name_or_path ${MODEL_NAME:- bert-large-uncased-whole-word-masking-finetuned-squad}"
ENTRYPOINT_ARGS+=" --dataset_name ${DATASET_NAME:- squad}"
ENTRYPOINT_ARGS+="${TASK_NAME:+ --task_name $TASK_NAME}"
ENTRYPOINT_ARGS+=" --do_train ${DO_TRAIN:- True}"
ENTRYPOINT_ARGS+=" --do_eval ${DO_EVAL:- True}"
ENTRYPOINT_ARGS+=" --max_seq_length ${MAX_SEQ_LENGTH:- 256}"
ENTRYPOINT_ARGS+="${MAX_TRAIN_SAMPLES:+ --max_train_samples $MAX_TRAIN_SAMPLES}"
ENTRYPOINT_ARGS+="${MAX_EVAL_SAMPLES:+ --max_eval_samples $MAX_EVAL_SAMPLES}"
ENTRYPOINT_ARGS+=" --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE:- 3}"
ENTRYPOINT_ARGS+=" --learning_rate ${LEARNING_RATE:- 3e-5}"
ENTRYPOINT_ARGS+=" --num_train_epochs ${NUM_TRAIN_EPOCHS:- 2}"
ENTRYPOINT_ARGS+=" --output_dir ${OUTPUT_DIR:- /home/training/output/bert_finetuned_model}"
ENTRYPOINT_ARGS+=" --overwrite_output_dir ${OVERWRITE_OUTPUT_DIR:- True}"
ENTRYPOINT_ARGS+="${EXTRAS:+ $EXTRAS}"

python3 "${TRAINING_FILE:-/home/training/training_scripts/run_qa.py}"$ENTRYPOINT_ARGS | tee /home/training/logs.txt
