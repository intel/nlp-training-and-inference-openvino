# -------------------------------------------------------------------------
# Copyright (C) 2022, Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Major Portions of this code are copyright of their respective authors and released under the Apache License Version 2.0:- openvino_contrib.
# For licensing see https://github.com/openvinotoolkit/openvino_contrib/blob/optimum-adapters/LICENSE
# -------------------------------------------------------------------------

from transformers import AutoTokenizer, AutoConfig
from optimum.intel.openvino import OVAutoModelForQuestionAnswering
from argparse import ArgumentParser
import time
import sys
import os
import numpy as np


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument("-mn", "--modelname", default='bert-large-uncased-whole-word-masking-finetuned-squad',
                      help="Name of the model", required=True
                      )
    args.add_argument("-mp", "--modelpath",
                      help="Required. Path to an .xml file with a trained model ", required=True
                      )
    args.add_argument("-mt", "--modeltype", help="Model type as pytorch or OV",
                      required=True)
    args.add_argument("-ar", "--adapter", help="Adapter OVMS or Openvino",
                      required=True)
    args.add_argument("-ni", "--iterations", type=int,
                      default=100, help="Number of Iterations")
    args.add_argument("-ip", "--inputpath",
                      default='/home/inference/inputs/input.txt', help="Path to the input file")
    args.add_argument("-cp", "--contextpath",
                      default='/home/inference/inputs/context.txt', help="Path to the input file")

    return parser


args = build_argparser().parse_args()
if os.path.exists(args.inputpath):
    with open(args.inputpath, 'r') as inputfile:
        question = inputfile.read()
else:
    sys.exit("Input file not found.")
if os.path.exists(args.contextpath):
    with open(args.contextpath, 'r') as inputfile:
        context = inputfile.read()
else:
    sys.exit("Context file not found.")


# Load model from HuggingFace Hub
config = AutoConfig.from_pretrained(args.modelname)
tokenizer = AutoTokenizer.from_pretrained(args.modelname)
if args.modeltype == 'pt' and args.adapter == 'openvino':
    model = OVAutoModelForQuestionAnswering.from_pretrained(
        args.modelpath, from_pt=True)
elif args.modeltype == 'ov' and args.adapter == 'openvino':
    model = OVAutoModelForQuestionAnswering.from_pretrained(
        args.modelpath, config=config)
elif args.modeltype == 'ov' and args.adapter == 'ovms':
    model = OVAutoModelForQuestionAnswering.from_pretrained(
        args.modelpath, inference_backend="ovms", config=config
    )
else:
    sys.exit("Model Type not found")
rows = []
input = tokenizer.encode_plus(
    question, context, return_tensors="np", add_special_tokens=True
)

# warmup step
model(**input, return_dict=True)
num_runs = args.iterations
start = time.time()
for _ in range(num_runs):
    result = model(**input, return_dict=True)
average_time = ((time.time() - start) * 1e3)/num_runs
print(f"Average inference time: {average_time}")
answer_start_scores = result["start_logits"]
answer_end_scores = result["end_logits"]

input_ids = input["input_ids"].tolist()[0]
answer_start = np.argmax(answer_start_scores)
answer_end = np.argmax(answer_end_scores) + 1
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
)
print(f"Question: {question}")
print(f"Answer: {answer}")
