
## Running Inference on OpenVINO™ Model Server - BERT Model 
Running the finetuned model in [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server)

1) Crate a folder structure as below
    | -bert
       | -1
2) keep your IR files in the bert/1/ directory and rename it to openvino_model.xml and openvino_model.bin
3)Start the ovms server by following command.(make sure to change the input shape as per the training)  
   docker run  --rm -p 9000:9000 -v $PWD/bert:/bert openvino/model_server:latest --model_name bert --model_path /bert --port 9000 --shape "{\"input_ids\": \"(1,256)\", \"attention_mask\": \"(1,256)\", \"token_type_ids\": \"(1,256)\"}" --log_level DEBUG --file_system_poll_wait_seconds 0



4) Create and activate a virtual enviornment and set all below env variables  
export MODEL_PATH=<IP_ADDRESS_OMVSSERVER>:9000/models/bert  
export VOCAB_FILE=path to vocab.txt file  
export INPUT="https://en.wikipedia.org/wiki/Bert_(Sesame_Street)"  
### **Note**:
Replace the IP with the IP of the ovms server container. (Use "docker inspect" to get the IP address)
Relpace the Vocab file with the one used during training
Replace the INPUT URL with any url of your choice to ask question related to it.



5) Run the shell script
chmod +x run_ov_client.sh
./run_ov_client.sh

