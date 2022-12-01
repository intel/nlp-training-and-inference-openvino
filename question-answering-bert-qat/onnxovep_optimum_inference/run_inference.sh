#!/bin/bash
python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir transformers onnxruntime-openvino torch pandas
ENTRYPOINT_ARGS=""
echo $INFERENCE_SCRIPT
if [ -f "$INFERENCE_SCRIPT" ]; then
    ENTRYPOINT_ARGS+=" --modelname ${MODEL_NAME:- bert-large-uncased-whole-word-masking-finetuned-squad}"
    ENTRYPOINT_ARGS+="${ONNX_MODEL_PATH:+ --modelpath $ONNX_MODEL_PATH}"
    ENTRYPOINT_ARGS+="${MODEL_TYPE:+ --modeltype $MODEL_TYPE}"
    ENTRYPOINT_ARGS+="${DEVICE:+ --device $DEVICE}"
    ENTRYPOINT_ARGS+="${NITER:+ --niter $NITER}"
    ENTRYPOINT_ARGS+="${MAX_SEQ_LENGTH:+ --max_seq_length $MAX_SEQ_LENGTH}"
    ENTRYPOINT_ARGS+=" --inputpath ${INPUT_PATH:- /home/inference/data/input.csv}"
    ENTRYPOINT_ARGS+=" --outputpath ${OUTPUT_PATH:- /home/inference/data/output.csv}"
    python3 $INFERENCE_SCRIPT$ENTRYPOINT_ARGS | tee /home/inference/logs.txt
    sleep 10
else
    echo 'Please pass the inference script.'
fi
