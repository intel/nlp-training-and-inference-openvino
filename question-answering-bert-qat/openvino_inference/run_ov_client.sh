#!/bin/bash

if [ ! -d "${PWD}/open_model_zoo" ]; then
  # Cloning the open_model_zoo branch to run the question answering demo.
  git clone https://github.com/openvinotoolkit/open_model_zoo.git
fi

# Changing the directory to the sample script path
cd open_model_zoo/demos/bert_question_answering_demo/python/

# Installation
pip install -U pip
pip install openvino-dev==2022.2.0
pip install openvino==2022.2.0
pip install ../../common/python[ovms] # Installing Python* Model API package from OpenVINO open-model-zoo

# Accepting arguments through enviornment variables
ENTRYPOINT_ARGS=""
ENTRYPOINT_ARGS+="${VOCAB_FILE:+ --vocab $VOCAB_FILE}"
ENTRYPOINT_ARGS+="${MODEL_PATH:+ --model $MODEL_PATH}"
ENTRYPOINT_ARGS+="${INPUT:+ --input $INPUT}"
ENTRYPOINT_ARGS+=" --adapter ${ADAPTER:- ovms}"
ENTRYPOINT_ARGS+=" --input_names ${INPUT_NAMES:- input_ids,attention_mask,token_type_ids}"
ENTRYPOINT_ARGS+=" --output_names ${OUTPUT_NAMES:- start_logits,end_logits}"
ENTRYPOINT_ARGS+=" -c"

# Executing the script with necessary arguments
python3 bert_question_answering_demo.py $ENTRYPOINT_ARGS