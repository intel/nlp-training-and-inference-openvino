# **Docker Run Instructions**
This document details instructions on how to run the onnxruntime inference container locally

## Project Structure 
```
├──data - input files to the inference
  ├──input.csv - Context and Question is passed in the csv file
  ├──output.csv  - Output is store in output.csv file
├──bert_inference_optimum_ort_ovep.py - Optimum ORT inference script
├──run_onnx_inference.sh - entrypoint for Optimum ORT inference docker
├──bert_inference_onnxruntime.py - Onnxruntime inference script
├──run_inference.sh - entrypoint for Onnxruntime inference docker
```


## Command to Run the docker image:  
Run the command below in the root of the application.  
```
   cd onnxovep_optimum_inference
```

### Docker run command to run INT8 ONNX file on OpenVINOExecutionProvider (Optimum ORT Script):  

`docker run -it --entrypoint /bin/bash --env PROVIDER=OpenVINOExecutionProvider --env NITER=10 --env ONNX_MODEL_PATH=/home/inference/models -v $(pwd)/../quantization_aware_training/models/bert_int8:/home/inference/models -v $(pwd):/home/inference --env INFERENCE_SCRIPT=/home/inference/bert_inference_optimum_ort_ovep.py openvino/ubuntu20_dev:2022.2.0 -c "/home/inference/run_onnx_inference.sh"`


### Docker run command to run INT8 ONNX file(generated after finetuning on the train container) on CPU Execution Provider(Optimum ORT Script):  

`docker run -it --entrypoint /bin/bash --env PROVIDER=CPUExecutionProvider --env NITER=10 --env ONNX_MODEL_PATH=/home/inference/models -v $(pwd)/../quantization_aware_training/models/bert_int8:/home/inference/models -v $(pwd):/home/inference --env INFERENCE_SCRIPT=/home/inference/bert_inference_optimum_ort_ovep.py openvino/ubuntu20_dev:2022.2.0 -c "/home/inference/run_onnx_inference.sh"`

## Docker run command to run INT8 ONNX file on the CPU with OpenVINO™ Runtime:  
`docker run -it --entrypoint /bin/bash --rm --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb -v $(pwd):/home/inference -v $(pwd)/../quantization_aware_training/models/bert_int8/:/home/inference/models --user 1000:1000 --env INFERENCE_SCRIPT=/home/inference/bert_inference_onnxruntime.py --env NITER=10 --env ONNX_MODEL_PATH=/home/inference/models/model.onnx --env DEVICE=CPU_FP32 --env MODEL_TYPE=INT8 --env MAX_SEQ_LENGTH=256 openvino/onnxruntime_ep_ubuntu18:2022.1.0 -c "/home/inference/run_inference.sh"`

## Docker run command to run INT8 ONNX file(generated after finetuning on the train container) on default CPU:  

`docker run -it --entrypoint /bin/bash --rm --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb -v $(pwd):/home/inference -v $(pwd)/../quantization_aware_training/models/bert_int8/:/home/inference/models --user 1000:1000 --env INFERENCE_SCRIPT=/home/inference/bert_inference_onnxruntime.py --env NITER=10 --env ONNX_MODEL_PATH=/home/inference/models/model.onnx --env DEVICE=cpu --env MODEL_TYPE=INT8 --env MAX_SEQ_LENGTH=256 openvino/onnxruntime_ep_ubuntu18:2022.1.0 -c "/home/inference/run_inference.sh"`  

## Summary:  
 The **INT8 ONNX file** generated after training in quantization_aware_training folder is **quantized and finetuned** using our **training container** and shows the **better performance** compared to **FP32 ONNX file**.   

### **Notes**:  
### **Volume Mount Model Folder**
`$(pwd)/models:/home/inference/models`  
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
     
  `docker run -it --entrypoint /bin/bash --env PROVIDER=OpenVINOExecutionProvider --env NITER=10 --env ONNX_MODEL_PATH=/home/inference/models -v $(pwd)/../quantization_aware_training/models/bert_int8:/home/inference/models -v $(pwd):/home/inference --env INFERENCE_SCRIPT=/home/inference/bert_inference_optimum_ort_ovep.py openvino/ubuntu20_dev:2022.2.0 -c '/home/inference/run_onnx_inference.sh "Where do I live?" "My name is Sarah and I live in London"'
  `

