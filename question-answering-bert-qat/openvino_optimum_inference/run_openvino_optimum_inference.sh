#!/bin/bash
EXTRAS="$*"
ENTRYPOINT_ARGS=""
echo $INFERENCE_SCRIPT
echo $MODEL_NAME

if [ -f "$INFERENCE_SCRIPT" ]; then
    ENTRYPOINT_ARGS+="${MODEL_NAME:+ --modelname $MODEL_NAME}"
    ENTRYPOINT_ARGS+="${MODEL_PATH:+ --modelpath $MODEL_PATH}"
    ENTRYPOINT_ARGS+="${MODEL_TYPE:+ --modeltype $MODEL_TYPE}"
    ENTRYPOINT_ARGS+="${ADAPTER:+ --adapter $ADAPTER}"
    ENTRYPOINT_ARGS+="${ITERATIONS:+ --iterations $ITERATIONS}"
    ENTRYPOINT_ARGS+="${EXTRAS:+ $EXTRAS}"
    python $INFERENCE_SCRIPT$ENTRYPOINT_ARGS | tee /home/inference/logs.txt
else
    echo 'Please pass the inference script.'
fi
