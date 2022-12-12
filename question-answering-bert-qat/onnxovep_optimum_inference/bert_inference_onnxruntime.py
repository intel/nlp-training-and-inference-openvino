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
import sys
from argparse import ArgumentParser
import pathlib
import pandas as pd

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


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument("-mn", "--modelname", default='bert-large-uncased-whole-word-masking-finetuned-squad',
                      help="Name of the model")
    args.add_argument("-mp", "--modelpath",
                      help="Path to the onnx model")
    args.add_argument("-mt", "--modeltype", help="INT8 or FP32")
    args.add_argument("-d", "--device", help="CPU or CPU_FP32")
    args.add_argument("-ni", "--niter", type=int,
                      default=10, help="Number of Iterations")
    args.add_argument("-msl", "--max_seq_length", type=int,
                      default=256, help="Max Sequence Length")
    args.add_argument("-ip", "--inputpath",
                      default='/home/inference/data/input.csv', help="Path to the input file")
    args.add_argument("-op", "--outputpath",
                      default='/home/inference/data/output.csv', help="Path to the output file")
    args.add_argument("-q", "--question",type=str,
                      help="Question")
    args.add_argument("-c", "--context",type=str,
                      help="Context")

    return parser

# Perform the one-off intialisation for the prediction. The init code is run once when the endpoint is setup.


def init():
    global tokenizer, onnx_session, torch_model, input_args
    input_args = build_argparser().parse_args()
    model_name = input_args.modelname
    # Create the tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

    model_path = input_args.modelpath
    device = input_args.device
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
def run_pytorch(question,context):
    logging.info("Question:", question)
    logging.info("Context: ", context)
    max_seq_length = int(input_args.max_seq_length)
    input_ids, input_mask, segment_ids, tokens = preprocess(
        question, context)
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

    n_iter = int(input_args.niter)
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
def run_onnx(question,context):
    logging.info("Question:", question)
    logging.info("Context: ", context)

    # Preprocess the question and context into tokenized ids
    input_ids, input_mask, segment_ids, tokens = preprocess(
        question, context)

    # Format the inputs for ONNX Runtime
    max_seq_length = int(input_args.max_seq_length)
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

    n_iter = int(input_args.niter)

    if input_args.modeltype == 'FP32':
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

    elif input_args.modeltype == 'INT8':
        # INT8 onnx file inputs
        model_inputs = {
            'input_ids':   [input_ids],
            'attention_mask':  [input_mask],
            'token_type_ids': [segment_ids]
        }

        # warm up step
        onnx_session.run(['start_logits', 'end_logits'], model_inputs)

        start = time.time()
        for _ in range(n_iter):
            outputs = onnx_session.run(['start_logits', 'end_logits'], model_inputs)
        average_time = (((time.time() - start)/n_iter) * 1e3)

    print(f"{input_args.modeltype} Average inference time using openvino-onnxruntime with {input_args.device} Execution provider: {average_time}")
    logging.info("Post-processing")

    # Post process the output of the model into an answer (or an error if the question could not be answered)
    results = postprocess(tokens, outputs[0], outputs[1])
    logging.info(results)

    return results


def run_inference():
    input_data=[]
    if input_args.question and input_args.context:
        input_data.append([input_args.context,input_args.question])
    elif os.path.exists(input_args.inputpath):
        if pathlib.Path(input_args.inputpath).suffix =='.csv':
            csv_data = pd.read_csv(input_args.inputpath)
            input_data = [row for row in csv_data.values]
        else:
            sys.exit('Input file extension is not supported')
    else:
        sys.exit("Input not found.")
    rows=[]
    for context,question in input_data:
        if torch_model:
            if input_args.niter and input_args.max_seq_length:
                results=run_pytorch(question,context)
                print('Question:',question)
                print('Answer:',results['answer'])
            else:
                print(
                    "Some Arguments['NITER','MAX_SEQ_LENGTH']  are not set. Exiting")
                exit(0)
        elif onnx_session:
            if input_args.modeltype and input_args.device and input_args.niter and input_args.max_seq_length:
                results=run_onnx(question,context)
                print('Question:',question)
                print('Answer:',results['answer'])
            else:
                print(
                    "Some Arguments['MODEL_TYPE', 'DEVICE', 'NITER', 'MAX_SEQ_LENGTH'] are not set. Exiting...")
                exit(0)
        rows.append([context,question,results['answer']])
    headers = ['Context','Question', 'Answer']
    answers = pd.DataFrame(rows, columns=headers)
    if os.path.exists(os.path.dirname(input_args.outputpath)) and pathlib.Path(input_args.outputpath).suffix =='.csv':
        answers.to_csv(input_args.outputpath, index=False)
        print('Results is stored in Output CSV file')
    else:
        print('Output is not stored in Output CSV file as ouputpath not exists')


if __name__ == '__main__':
    tokenizer = onnx_session = torch_model = input_args = None
    init()
    run_inference()