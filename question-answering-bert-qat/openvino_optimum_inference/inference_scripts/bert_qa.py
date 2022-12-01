# -------------------------------------------------------------------------
# Copyright (C) 2022, Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Major Portions of this code are copyright of their respective authors and released under the Apache License Version 2.0:-
# For licensing see https://github.com/huggingface/optimum-intel/blob/main/LICENSE
# -------------------------------------------------------------------------

from transformers import AutoTokenizer, AutoConfig,pipeline
from optimum.intel.openvino import OVModelForQuestionAnswering
from argparse import ArgumentParser
import time
import sys
import os
import numpy as np
import pathlib
import pandas as pd

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
    args.add_argument("-ni", "--iterations", type=int,
                      default=100, help="Number of Iterations")
    args.add_argument("-ip", "--inputpath",
                      default='/home/inference/data/input.csv', help="Path to the input file")
    args.add_argument("-op", "--outputpath",
                      default='/home/inference/data/output.csv', help="Path to the output file")

    return parser


args = build_argparser().parse_args()
if os.path.exists(args.inputpath):
    if pathlib.Path(args.inputpath).suffix =='.csv':
        csv_data = pd.read_csv(args.inputpath)
        input_data = [row for row in csv_data.values]
    else:
        sys.exit('Input file extension is not supported')
else:
    sys.exit("Input file not found.")


# Load model from HuggingFace Hub

tokenizer = AutoTokenizer.from_pretrained(args.modelname)
if args.modeltype == 'pt':
    model = OVModelForQuestionAnswering.from_pretrained(
        args.modelpath,from_transformers=True)
elif args.modeltype == 'ov':
    model = OVModelForQuestionAnswering.from_pretrained(
        args.modelpath)
else:
    sys.exit("Model Type not found")

pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
rows = []
for context,question in input_data:
    # warmup step
    outputs = pipe(question,context)
    num_runs = args.iterations
    start = time.time()
    for _ in range(num_runs):
        outputs = pipe(question,context)
    average_time = ((time.time() - start) * 1e3)/num_runs
    print(f"Question: {question}")
    print(f"Context: {context}")
    print("Answer",outputs['answer'])
    print(f"Average inference time: {average_time}")
    rows.append([context,question,outputs['answer']])
headers = ['Context','Question', 'Answer']
answers = pd.DataFrame(rows, columns=headers)
answers.to_csv(args.outputpath, index=False)
print('Results is stored in Output CSV file')