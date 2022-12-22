# Docker Run Instructions
This document details instructions on how to run the inference container locally

## Project Structure 
```
├──data - input files to the inference
  ├──input.csv - Context and Question is passed in the csv file
  ├──output.csv  - Output is store in output.csv file
├──run_openvino_optimum_inference.sh - entrypoint to the inference container 
├──inference_scripts 
  ├──bert_qa.py - Inference script for BERT squad use case
 
```

## Docker Run Command
Update the  `<openvino_model.xml_directory>`  with respective path e.g., `/home/inference/models`  directory containing xmls
    
   ```
   cd openvino_optimum_inference
    
   docker run -it --entrypoint /bin/bash --env MODEL_NAME=bert-large-uncased-whole-word-masking-finetuned-squad --env MODEL_PATH=<openvino_model.xml_directory> --env MODEL_TYPE=ov --env ITERATIONS=100 --env INFERENCE_SCRIPT=/home/inference/inference_scripts/bert_qa.py -v  $(pwd)/../quantization_aware_training/models/bert_int8:/home/inference/models -v $(pwd):/home/inference openvino/ubuntu20_dev:2022.2.0 -c "/home/inference/run_openvino_optimum_inference.sh"
   ```
### **Notes**:  
Please find the volume mount and environment variable details below:

#### **Volume Mount Model Folder**
`$(pwd)/../quantization_aware_training/models/bert_int8:/home/inference/models`  
It will mount the host system model folder to container model folder.

#### **Volume mount Input Folder**
`$(pwd)/inputs:/home/inference/inputs`  
It will mount the host system ‘inputs’ folder to container inputs folder.
Context and Question can be mentioned in input.csv file under data folder. We have passed input.csv file as default.

#### **Input Data**
We have options to run inference on single and multiple inputs.  
- In case of multiple input, we need to provide input csv file. 
This file will be read and inference will be performed and the corresponding outputs will be saved as outputs.csv.  
- In case of single input, the variables context and question are used to facilitate this.  
  These variables are passed as an argument to the inference script. If they are empty strings or not passed as argument, then the default behaviour is to read the input csv file and run inference. If we want to try with question and context variable, use the below command
     ```  
     docker run -it --entrypoint /bin/bash --env MODEL_NAME=bert-large-uncased-whole-word-masking-finetuned-squad --env MODEL_PATH=/home/inference/models --env MODEL_TYPE=ov --env ITERATIONS=100 --env INFERENCE_SCRIPT=/home/inference/inference_scripts/bert_qa.py -v  $(pwd)/../quantization_aware_training/models/bert_int8:/home/inference/models -v $(pwd):/home/inference openvino/ubuntu20_dev:2022.2.0 -c '/home/inference/run_openvino_optimum_inference.sh "Where do I live?" "My name is Sarah and I live in London"'  
     ```

#### **Model Type:**
Pass env variable MODEL_TYPE= pt inorder to load Pytorch file.
Pass MODEL_TYPE= ov inorder to load IR file.

#### **Number of iterations**
We can use ITERATIONS env variable to update number of iterations.

