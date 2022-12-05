#!/bin/bash
python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir git+https://github.com/huggingface/optimum-intel.git@v1.5.2#egg=optimum-intel[openvino]
export PATH=$PATH:/home/openvino/.local/lib:/home/openvino/.local/bin

ENTRYPOINT_ARGS=""

if [ -f "$INFERENCE_SCRIPT" ]; then
    ENTRYPOINT_ARGS+="${MODEL_NAME:+ --modelname $MODEL_NAME}"
    ENTRYPOINT_ARGS+="${MODEL_PATH:+ --modelpath $MODEL_PATH}"
    ENTRYPOINT_ARGS+="${MODEL_TYPE:+ --modeltype $MODEL_TYPE}"
    ENTRYPOINT_ARGS+="${ITERATIONS:+ --iterations $ITERATIONS}"
    ENTRYPOINT_ARGS+=" --inputpath ${INPUT_PATH:- /home/inference/data/input.csv}"
    ENTRYPOINT_ARGS+=" --outputpath ${OUTPUT_PATH:- /home/inference/data/output.csv}"
    python3 $INFERENCE_SCRIPT$ENTRYPOINT_ARGS --question "$1" --context "$2" | tee /home/inference/logs.txt
else
    echo 'Please pass the inference script.'
fi
