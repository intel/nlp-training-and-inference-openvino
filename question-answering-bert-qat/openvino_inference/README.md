
## Running Inference on OpenVINO™ Model Server - BERT Model 
Running the finetuned model in [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server)

1) Create a folder structure as below
    | -bert
       | -1  
2) keep your IR files in the bert/1/ directory and start the ovms server
3)Start the ovms server using following command.(make sure to change the input shape as per the training)    
   `docker run  --rm -p 9000:9000 -v $PWD/bert:/bert openvino/model_server:latest --model_name bert --model_path /bert --port 9000 --shape "{\"input_ids\": \"(1,256)\", \"attention_mask\": \"(1,256)\", \"token_type_ids\": \"(1,256)\"}" --log_level DEBUG --file_system_poll_wait_seconds 0`



4) Run Inference on ovms server using below command. It will download inference script from open_model_zoo and run inference on ovms server:  
`cd <gitrepofolder>/openvino_inference`  

`docker run -it --entrypoint /bin/bash -v "$(pwd)":/home/inference -v "$(pwd)"/../quantization_aware_training/models/bert_int8/vocab.txt:/home/inference/vocab.txt --env VOCAB_FILE=/home/inference/vocab.txt --env  INPUT="https://en.wikipedia.org/wiki/Bert_(Sesame_Street)" --env MODEL_PATH=<IP ADDRESS OVMS SERVER>:9000/models/bert openvino/ubuntu20_dev:2022.2.0  -c /home/inference/run_ov_client.sh`


### **Note**:
Replace the IP with the IP of the ovms server container. (Use "docker inspect <ovms container>" to get the IP address)  
Vocab file is available in the model folder.  
Replace the INPUT URL with any url of your choice to ask question related to it.


