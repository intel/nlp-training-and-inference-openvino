#!/bin/bash
python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir transformers onnxruntime-openvino torch
EXTRAS="$*"
ENTRYPOINT_ARGS=""
echo $INFERENCE_SCRIPT
if [ -f "$INFERENCE_SCRIPT" ]; then
    
    python3 $INFERENCE_SCRIPT | tee /home/inference/logs.txt
    sleep 10
else
    echo 'Please pass the inference script.'
fi
