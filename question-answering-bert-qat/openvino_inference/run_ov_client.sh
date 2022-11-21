#!/bin/bash
python3 -m pip install --upgrade pip
python3 -m pip install https://github.com/openvinotoolkit/open_model_zoo/demos/common/python

EXTRAS="$*"
ENTRYPOINT_ARGS=""
echo $INFERENCE_SCRIPT
#if [ -f "$INFERENCE_SCRIPT" ]; then
    
 #   python3 $INFERENCE_SCRIPT | tee /home/inference/logs.txt
  #  sleep 10
#else
 #   echo 'Please pass the inference script.'
#fi
python3 bert_question_answering_demo.py
            --vocab=<models_dir>/models/intel/bert-small-uncased-whole-word-masking-squad-0001/vocab.txt
            --model=localhost:9000/models/bert
            --input_names="input_ids,attention_mask,token_type_ids"
            --output_names="output_s,output_e"
            --input="https://en.wikipedia.org/wiki/Bert_(Sesame_Street)"
            --adapter ovms
            -c