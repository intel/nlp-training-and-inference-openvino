from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForQuestionAnswering
import torch
import numpy as np
import time
import sys
from argparse import ArgumentParser
import os
import pathlib
import pandas as pd

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument("-mn", "--modelname", default='bert-large-uncased-whole-word-masking-finetuned-squad',
                      help="Name of the model")
    args.add_argument("-mp", "--modelpath",
                      help="Path to the onnx model")
    args.add_argument("-p", "--provider", help="CPUExecutionProvider or OpenVINOExecutionProvider")
    args.add_argument("-ni", "--niter", type=int,
                      default=10, help="Number of Iterations")
    args.add_argument("-ip", "--inputpath",
                      default='/home/inference/data/input.csv', help="Path to the input file")
    args.add_argument("-op", "--outputpath",
                      default='/home/inference/data/output.csv', help="Path to the output file")
    args.add_argument("-q", "--question",type=str,
                      help="Question")
    args.add_argument("-c", "--context",type=str,
                      help="Context")
    return parser

def run_inference():
    args = build_argparser().parse_args()
    input_data=[]
    if args.question and args.context:
        input_data.append([args.context,args.question])

    elif os.path.exists(args.inputpath):
        if pathlib.Path(args.inputpath).suffix =='.csv':
            csv_data = pd.read_csv(args.inputpath)
            input_data = [row for row in csv_data.values]
        else:
            sys.exit('Input file extension is not supported')
    else:
        sys.exit("Input file not found.")

    tokenizer = AutoTokenizer.from_pretrained(args.modelname)

    if args.provider in ['OpenVINOExecutionProvider','CPUExecutionProvider']:
        print(
            "Inference on " + str(args.provider))
        model = ORTModelForQuestionAnswering.from_pretrained(args.modelpath,provider=args.provider)
    else:
        print("Provider " + str(args.provider)+ " is not supported. Exiting...")
        exit(0)
    rows=[]
    for context,question in input_data:
        inputs = tokenizer(question, context, return_tensors="pt")
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])
        #Warmup Step
        outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
        start = time.time()
        num_runs=args.niter
        for _ in range(num_runs):
            outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
        average_time = ((time.time() - start) * 1e3)/num_runs
        
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        input = tokenizer.encode_plus(
            question, context, return_tensors="np", add_special_tokens=True
        )
        input_ids = input["input_ids"].tolist()[0]
        answer_start = np.argmax(start_scores)
        answer_end = np.argmax(end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
        )
        print(f"Question: {question}")
        print(f"Answer: {answer}")  
        print(f"Average inference time: {average_time}")
        rows.append([context,question,answer])
    headers = ['Context','Question', 'Answer']
    answers = pd.DataFrame(rows, columns=headers)
    answers.to_csv(args.outputpath, index=False)
    print('Results is stored in Output CSV file')

if __name__ == '__main__':
    run_inference()