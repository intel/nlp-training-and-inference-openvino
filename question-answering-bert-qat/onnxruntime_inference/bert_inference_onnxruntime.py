"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

# ==============================================================================
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

This script is made with the reference of example scripts mentioned on the below links
https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/azureml/Inference_Bert_with_OnnxRuntime_on_AzureML.ipynb
https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/OpenVINO_EP/yolov4_object_detection/yolov4.py
"""

import os
import logging
import json
import numpy as np
import onnxruntime
import transformers
import torch
import time

# The pre process function take a question and a context, and generates the tensor inputs to the model:
# - input_ids: the words in the question encoded as integers
# - attention_mask: not used in this model
# - token_type_ids: a list of 0s and 1s that distinguish between the words of the question and the words of the context
# This function also returns the words contained in the question and the context, so that the answer can be decoded into a phrase.


def preprocess(question, context):
    encoded_input = tokenizer(question, context)
    tokens = tokenizer.convert_ids_to_tokens(encoded_input.input_ids)
    return (encoded_input.input_ids, encoded_input.attention_mask, encoded_input.token_type_ids, tokens)

# The post process function takes the list of tokens in the question and context, as well as the output of the
# model, the list of log probabilities for the choices of start and end of the answer, and maps it back to an
# answer to the question that is asked of the context.


def postprocess(tokens, start, end):
    results = {}
    answer_start = np.argmax(start)
    answer_end = np.argmax(end)
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
        results['answer'] = answer.capitalize()
    else:
        results['error'] = "I am unable to find the answer to this question. Can you please ask another question?"
    return results

# Perform the one-off intialisation for the prediction. The init code is run once when the endpoint is setup.


def init():
    global tokenizer, onnx_session, torch_model

    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    # Create the tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

    model_path = os.getenv('ONNX_MODEL_PATH')
    device = os.getenv('DEVICE')
    if model_path:
        # Create an ONNX Runtime session to run the ONNX model
        if device == 'cpu':
            print(
                "Device type selected is 'cpu' which is the default CPU Execution Provider (MLAS)")
            onnx_session = onnxruntime.InferenceSession(
                model_path, providers=["CPUExecutionProvider"])
        elif device == 'CPU_FP32':
            onnx_session = onnxruntime.InferenceSession(model_path, providers=[
                                                   'OpenVINOExecutionProvider'], provider_options=[{'device_type': device}])
            print("Device type selected is: " + device +
                  " using the OpenVINO Execution Provider")
            '''
            other 'device' options are: (Any hardware target can be assigned if you have the access to it)
            'CPU_FP32', 'GPU_FP32', 'GPU_FP16', 'MYRIAD_FP16', 'VAD-M_FP16'
            '''
        else:
            print("Either device is not set or not supported. Exiting...")
            exit(0)
    else:
        torch_model = transformers.BertForQuestionAnswering.from_pretrained(
            model_name)


# Run the PyTorch model, for functional and performance comparison
def run_pytorch(raw_data):
    inputs = json.loads(raw_data)

    logging.info("Question:", inputs["question"])
    logging.info("Context: ", inputs["context"])
    max_seq_length = int(os.getenv('MAX_SEQ_LENGTH'))
    input_ids, input_mask, segment_ids, tokens = preprocess(
        inputs["question"], inputs["context"])
    if len(input_ids) < max_seq_length:
        input_ids.extend([0]*(max_seq_length-len(input_ids)))
    else:
        input_ids = input_ids[:max_seq_length]
    if len(input_mask) < max_seq_length:
        input_mask.extend([0]*(max_seq_length-len(input_mask)))
    else:
        input_mask = input_mask[:max_seq_length]
    if len(segment_ids) < max_seq_length:
        segment_ids.extend([0]*(max_seq_length-len(segment_ids)))
    else:
        segment_ids = segment_ids[:max_seq_length]

    n_iter = int(os.getenv('NITER'))
    # warmup step
    torch_model(torch.tensor([input_ids]),
          token_type_ids=torch.tensor([segment_ids]))

    start = time.time()
    for _ in range(n_iter):
        model_outputs = torch_model(torch.tensor(
            [input_ids]),  token_type_ids=torch.tensor([segment_ids]))
    average_time = (((time.time() - start)/n_iter) * 1e3)
    print(f"Average inference time using pytorch: {average_time}")

    return postprocess(tokens, model_outputs.start_logits.detach().numpy(), model_outputs.end_logits.detach().numpy())

# Run the ONNX model with ONNX Runtime
def run_onnx(raw_data):

    inputs = json.loads(raw_data)

    logging.info("Question:", inputs["question"])
    logging.info("Context: ", inputs["context"])

    # Preprocess the question and context into tokenized ids
    input_ids, input_mask, segment_ids, tokens = preprocess(
        inputs["question"], inputs["context"])

    # Format the inputs for ONNX Runtime
    max_seq_length = int(os.getenv('MAX_SEQ_LENGTH'))
    if len(input_ids) < max_seq_length:
        input_ids.extend([0]*(max_seq_length-len(input_ids)))
    else:
        input_ids = input_ids[:max_seq_length]
    if len(input_mask) < max_seq_length:
        input_mask.extend([0]*(max_seq_length-len(input_mask)))
    else:
        input_mask = input_mask[:max_seq_length]
    if len(segment_ids) < max_seq_length:
        segment_ids.extend([0]*(max_seq_length-len(segment_ids)))
    else:
        segment_ids = segment_ids[:max_seq_length]

    n_iter = int(os.getenv('NITER'))

    if os.getenv('MODEL_TYPE') == 'FP32':
        model_inputs = {
            'input_ids':   [input_ids],
            'input_mask':  [input_mask],
            'segment_ids': [segment_ids]
        }

        # warm up step
        onnx_session.run(['start_logits', 'end_logits'], model_inputs)

        start = time.time()
        for _ in range(n_iter):
            outputs = onnx_session.run(['start_logits', 'end_logits'], model_inputs)
        average_time = (((time.time() - start)/n_iter) * 1e3)

    elif os.getenv('MODEL_TYPE') == 'INT8':
        # INT8 onnx file inputs
        model_inputs = {
            'input_ids':   [input_ids],
            'attention_mask':  [input_mask],
            'token_type_ids': [segment_ids]
        }

        # warm up step
        onnx_session.run(['output.0', 'output.1'], model_inputs)

        start = time.time()
        for _ in range(n_iter):
            outputs = onnx_session.run(['output.0', 'output.1'], model_inputs)
        average_time = (((time.time() - start)/n_iter) * 1e3)

    print(f"{os.getenv('MODEL_TYPE')} Average inference time using openvino-onnxruntime with {os.getenv('DEVICE')} Execution provider: {average_time}")
    logging.info("Post-processing")

    # Post process the output of the model into an answer (or an error if the question could not be answered)
    results = postprocess(tokens, outputs[0], outputs[1])
    logging.info(results)

    return results


def run_inference():
    input = "{\"question\": \"Which NFL team represented the NFC at Super Bowl 50?\", \"context\": \"Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the 'golden anniversary' with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as 'Super Bowl L'), so that the logo could prominently feature the Arabic numerals 50.\"}"
    if torch_model:
        if os.getenv('NITER') and os.getenv('MAX_SEQ_LENGTH'):
            print(run_pytorch(input))
        else:
            print(
                "Some Environment variables['NITER','MAX_SEQ_LENGTH']  are not set. Exiting")
            exit(0)
    elif onnx_session:
        if os.getenv('MODEL_TYPE') and os.getenv('DEVICE') and os.getenv('NITER') and os.getenv('MAX_SEQ_LENGTH'):
            print(run_onnx(input))
        else:
            print(
                "Some Environment Variables['MODEL_TYPE', 'DEVICE', 'NITER', 'MAX_SEQ_LENGTH'] are not set. Exiting...")
            exit(0)


if __name__ == '__main__':
    tokenizer = onnx_session = torch_model = None
    init()
    run_inference()
