# **Docker Run Instructions**
This document details instructions on how to run the onnxruntime inference container locally

## Project Structure 
```
├──bert_inference_onnxruntime.py - inference script
├──run-onnx-inference.sh - entrypoint for inference docker
├──torch_to_onnx.py - optional helper script to convert a pytorch model to onnx format
```



## Command to Run the docker image:  
Run the command below in the root of the application.  
```
   cd onnxruntime_inference
```

### Docker run command to do inference with the pytorch* model:  
    
```
  docker run -it --entrypoint /bin/bash --rm --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb -v $(pwd):/home/inference --user 1000:1000 --env NITER=10 --env INFERENCE_SCRIPT=/home/inference/bert_inference_onnxruntime.py --env MAX_SEQ_LENGTH=256 openvino/onnxruntime_ep_ubuntu18:2022.1.0 -c "/home/inference/run_onnx_inference.sh"
``` 


### Docker run command to do inference with the FP32 ONNX file(generated from pytorch* model) on default CPU:  

#### Instructions to download and convert to FP32 ONNX model

Please use below helper script. And it needs some dependencies to be installed
`
pip3 install torch transformers
python torch_to_onnx.py
`

`docker run -it --entrypoint /bin/bash --rm --device-cgroup-rule='c 189:* rmw' -v $(pwd):/home/inference --user 1000:1000 --env INFERENCE_SCRIPT=/home/inference/bert_inference_onnxruntime.py --env NITER=10 --env ONNX_MODEL_PATH=<FP32_ONNX_FILE> --env DEVICE=cpu --env MODEL_TYPE=FP32 --env MAX_SEQ_LENGTH=256 openvino/onnxruntime_ep_ubuntu18:2022.1.0 -c "/home/inference/run_onnx_inference.sh" `

### Docker run command  do inference with the FP32 ONNX file(generated from pytorch* model) on the CPU with OpenVINO™ Runtime:  

`docker run -it --entrypoint /bin/bash --rm --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb -v $(pwd):/home/inference --user 1000:1000 --env INFERENCE_SCRIPT=/home/inference/bert_inference_onnxruntime.py --env NITER=10 --env ONNX_MODEL_PATH=<FP32_ONNX_FILE> --env DEVICE=CPU_FP32 --env MODEL_TYPE=FP32 --env MAX_SEQ_LENGTH=256 openvino/onnxruntime_ep_ubuntu18:2022.1.0 -c "/home/inference/run_onnx_inference.sh"`

### Docker run command to run INT8 ONNX file(generated after finetuning on the train container) on default CPU:  

`docker run -it --entrypoint /bin/bash --rm --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb -v  $(pwd)/../quantization_aware_training/model/bert_int8:/home/inference/models -v $(pwd):/home/inference --user 1000:1000 --env INFERENCE_SCRIPT=/home/inference/bert_inference_onnxruntime.py --env NITER=10 --env ONNX_MODEL_PATH=/home/inference/models/ov_model.onnx --env DEVICE=cpu --env MODEL_TYPE=INT8 --env MAX_SEQ_LENGTH=256 openvino/onnxruntime_ep_ubuntu18:2022.1.0 -c "/home/inference/run_onnx_inference.sh"`

### Docker run command to run INT8 ONNX file on the CPU with OpenVINO™ Runtime:  

`docker run -it --entrypoint /bin/bash --rm --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb -v  $(pwd)/../quantization_aware_training/model/bert_int8:/home/inference/models -v $(pwd):/home/inference --user 1000:1000 --env INFERENCE_SCRIPT=/home/inference/bert_inference_onnxruntime.py --env NITER=10 --env ONNX_MODEL_PATH=/home/inference/models/ov_model.onnx --env DEVICE=CPU_FP32 --env MODEL_TYPE=INT8 --env MAX_SEQ_LENGTH=256 openvino/onnxruntime_ep_ubuntu18:2022.1.0 -c "/home/inference/run_onnx_inference.sh"
`


## Summary:  
 The **INT8 ONNX file** generated after training in quantization_aware_training folder is **quantized and finetuned** using our **training container** and shows the **better performance** compared to **FP32 ONNX file**.   


### **Note:**  
**'MAX_SEQ_LENGTH'** environment variable needs to be updated with the one used during training. Currently it is set to length **256** as the finetuned model stored in the **'model'** folder was trained with the sequence length of **256**.  
