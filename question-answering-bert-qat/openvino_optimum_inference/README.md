# Docker Run Instructions
This document details instructions on how to run the inference container locally

## Project Structure 
```
├──inputs - input files to the inference
  ├──context.txt - context file related to the use case
  ├──input.txt  - input Question
├──run-inference.sh - entrypoint to the inference container 
├──inference_scripts 
  ├──bert_qa.py - Inference script for BERT squad use case
 
```


## Docker Run Command
1. OpenVINO™ adapter:    
  
   Replace `<OV_model.xml_directory>` with respective path e.g., `/home/inference/models`  
   Replace `<registry>` with the local or private registry address. If using local registry, edit it to "localhost:5000"  
    
   ```
   cd openvino_optimum_inference
    
   docker run -it --entrypoint /bin/bash --env MODEL_NAME=bert-large-uncased-whole-word-masking-finetuned-squad --env MODEL_PATH=<OV_model.xml_directory> --env MODEL_TYPE=ov  --env ADAPTER=openvino --env ITERATIONS=100 --env INFERENCE_SCRIPT=/home/inference/inference_scripts/bert_qa.py -v  $(pwd)/../quantization_aware_training/model/bert_int8:/home/inference/models -v $(pwd):/home/inference <registry>/openvino_optimum -c "/home/inference/run_openvino_optimum_inference.sh"
   ```
2. OpenVINO™ Model Server adapter (Ensure OpenVINO™ Model Server is running, check with `docker ps`)     
   Get `<IP_ADDRESS_OMVSSERVER>` with following commands  
     
   `docker ps` to get the Container Id of OpenVINO™ Model Server  
   `docker inspect <container id>` to get the IP Addredd  
     
   ```
   cd openvino_optimum_inference

   docker run -it --entrypoint /bin/bash --env MODEL_NAME=bert-large-uncased-whole-word-masking-finetuned-squad --env MODEL_PATH=<IP_ADDRESS_OMVSSERVER>/models/bert --env MODEL_TYPE=ov  --env ADAPTER=ovms --env ITERATIONS=100 --env INFERENCE_SCRIPT=/home/inference/inference_scripts/bert_qa.py -v  $(pwd)/../quantization_aware_training/model/bert_int8:/home/inference/models -v $(pwd):/home/inference <registry>/openvino_optimum -c "/home/inference/run_openvino_optimum_inference.sh"
   ```

## Running OpenVINO™ Model Server - BERT Model 
```
cd <MODEL_PATH>    #MODEL_PATH: Location where the .xml and .bin files of the model are located i.e., cd till `bert_int8` folder  
mkdir -p bert/1  
cp mapping_config.json  ov_model.bin  ov_model.xml bert/1/  
```
  
```
docker run  --rm -p 9000:9000 -v $PWD/bert:/bert openvino/model_server:latest --model_name bert --model_path /bert --port 9000 --shape "{\"input_ids\": \"(-1,-1)\", \"attention_mask\": \"(-1,-1)\", \"token_type_ids\": \"(-1,-1)\"}" --log_level DEBUG --file_system_poll_wait_seconds 0 --plugin_config "{\"PERFORMANCE_HINT\":\"LATENCY\"}"
```

### **Notes**:  
Please find the volume mount and environment variable details below:

#### **Volume Mount Model Folder**
`$(pwd)/models:/home/inference/models`  
It will mount the host system model folder to container model folder.

#### **Volume mount Input Folder**
`$(pwd)/inputs:/home/inference/inputs`  
It will mount the host system ‘inputs’ folder to container inputs folder.
Question can be mentioned in input.txt file and context can be mentioned in context.txt file under inputs folder. We have passed these files as default value and we can update the input files(question and context file) by passing arguments after image name while running docker. Argument for question file path is inputpath. Argument for context file path is contextpath. 

#### **Volume Mount Inference Scripts**
`$(pwd)/inference_scripts:/home/inference/inference_scripts`     
It will mount the host system ‘inference_scripts’ folder to container inference_scripts folder.
Python Inference scripts file should be in inference_scripts folder.

#### **Model Type:**
Pass env variable MODEL_TYPE= pt inorder to load Pytorch file.
Pass MODEL_TYPE= ov inorder to load IR file.

#### **Number of iterations**
We can use ITERATIONS env variable to update number of iterations.

#### **Adapter**
If we need to do inference in OpenVINO™ Model Server pass env variable  ADAPTER =ovms else pass ADAPTER=openvino.

If we pass ADAPTER=openvino then modeltype can be ov or pt. In modelpath we can mention path of OpenVINO™ or Pytorch model.

If we pass ADAPTER=ovms then modeltype should be ov. In model path environment variable, we can mention OpenVINO™ Model Server running instance reference. For eg: env variable `MODEL_PATH=<hostname_ovms_server>:9000/models/bert`.
We can run OpenVINO™ Model Server by using following command:  
`docker run --rm -p 9000:9000 -v $PWD/bert:/bert openvino/model_server:latest --model_name bert --model_path /bert --port 9000 --shape "{\"input_ids\": \"(-1,-1)\", \"attention_mask\": \"(-1,-1)\", \"token_type_ids\": \"(-1,-1)\"}" --log_level DEBUG --file_system_poll_wait_seconds 0 --plugin_config "{\"PERFORMANCE_HINT\":\"LATENCY\"}"`

The above command should be triggered from the path containing OpenVINO™ model and model should be in the directory structure accepted by [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server).
In order to communicate inference container with OpenVINO™ Model Server container we need to get ip address of OpenVINO™ Model Server container and pass the same ip while running inference container. For eg: env variable `MODEL_PATH=<hostname_ovms_server>:<port_ovmserver>/models/bert`
