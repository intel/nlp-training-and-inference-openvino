#!/bin/bash

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
ENTRYPOINT_ARGS+=" --nncf_config ${NNCF_CONFIG:- /home/training/config/bert_config.json}"
ENTRYPOINT_ARGS+="${EXTRAS:+ $EXTRAS}"

python "${TRAINING_FILE:-/home/training/training_scripts/run_qa.py}"$ENTRYPOINT_ARGS | tee /home/training/logs.txt
