"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
This script is made with the reference to the sample mentioned in the below link

https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/azureml/Inference_Bert_with_OnnxRuntime_on_AzureML.ipynb
"""
# ==============================================================================
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import torch
from transformers import BertForQuestionAnswering

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
model_path = "./" + model_name + ".onnx"
model = BertForQuestionAnswering.from_pretrained(model_name)

# set the model to inference mode
# It is important to call torch_model.eval() or torch_model.train(False) before exporting the model,
# to turn the model to inference mode. This is required since operators like dropout or batchnorm
# behave differently in inference and training mode.
model.eval()

# Generate dummy inputs to the model. Adjust if neccessary
inputs = {
    # list of numerical ids for the tokenized text
    'input_ids':   torch.randint(32, [1, 32], dtype=torch.long),
    # dummy list of ones
    'attention_mask': torch.ones([1, 32], dtype=torch.long),
    # dummy list of ones
    'token_type_ids':  torch.ones([1, 32], dtype=torch.long)
}

symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
torch.onnx.export(model,                                         # model being run
                  (inputs['input_ids'],
                   inputs['attention_mask'],
                   inputs['token_type_ids']),                    # model input (or a tuple for multiple inputs)
                  # where to save the model (can be a file or file-like object)
                  model_path,
                  # the ONNX version to export the model to
                  opset_version=11,
                  # whether to execute constant folding for optimization
                  do_constant_folding=True,
                  input_names=['input_ids',
                               'input_mask',
                               'segment_ids'],                   # the model's input names
                  # the model's output names
                  output_names=['start_logits', "end_logits"],
                  dynamic_axes={'input_ids': symbolic_names,
                                'input_mask': symbolic_names,
                                'segment_ids': symbolic_names,
                                'start_logits': symbolic_names,
                                'end_logits': symbolic_names})   # variable length axes
