# Docker Run Instructions
This document details instructions on how to run the training container locally. 

## Project Structure 
```
├──config - quantization json file
├──model - output directory 
 ├──bert_int8 - output model will be saved in this folder
   ├──mapping_config.json - mapping the BERT model output nodes
   ├──vocab.txt - needed by the BERT model for the inference
├──training_scripts - scripts related to training pipeline
  ├──run_qa.py - training script using Hugging Face transformers API
  ├──trainer_qa.py - supporting functions for training pipeline
  ├──utils_qa.py - supporting functions for training pipeline
├──run-qat.sh - entrypoint to the training container
```

## Docker Run Command:  
1. On Intel* CPU:  
Here `registry_name` should be local or private registry address. If using local registry, edit it to "localhost:5000"  
  
    `cd quantization_aware_training`

    `docker run --network=host -v "$(pwd)":/home/training --env MAX_TRAIN_SAMPLES=50 --env MAX_EVAL_SAMPLES=50 <registry_name>/openvino_optimum --doc_stride 128 --save_strategy epoch`

2. On NVIDIA* GPU:  
    `cd quantization_aware_training`

     `docker run --network=host -v "$(pwd)":/home/training --env CUDA_HOME=/usr/local/cuda -v /usr/local/cuda:/usr/local/cuda --env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} --gpus 'all,capabilities=compute' <registry>/<registry_name>/openvino_optimum --doc_stride 128 --save_strategy epoch`

### **Note**:  
The following environment variables are passed as part of the docker run command. Currently some of these environment variables are set with the default values in the entrypoint script of the container. Default values are taken based on the finetuning experiments done previously.  

**FILE**: Training script(default is '/home/training/training_scripts/run_qa.py')  
**MODEL_NAME**: Name of the huggingface model.(default is 'bert-large-uncased-whole-word-masking-finetuned-squad')  
**DATASET_NAME**: Name of the dataset. (default is 'squad')  
**TASK_NAME**: Name of the task of the dataset.  
**DO_TRAIN**: Pass 'True' to enable the training. (default is 'True')  
**DO_EVAL**: Pass 'True' to enable the evaluation. (default is 'True' )  
**MAX_SEQ_LENGTH**: Maximum length accepted by the inputs. (default is 256)  
**MAX_TRAIN_SAMPLES**: Pass it to use a subset of the dataset during training.  
**MAX_EVAL_SAMPLES**: Pass it to use a subset of the dataset for evaluation.  
**PER_DEVICE_TRAIN_BATCH_SIZE**: Input batch size for training. (default is 3)  
**LEARNING_RATE**: Learning rate for training. (default is 3e-5)  
**NUM_TRAIN_EPOCHS**: Train epoch for training. (default is 2)  
**OUTPUT_DIR**: Directory to store the trained models. (default is '/home/training/output/bert_finetuned_model')  
**OVERWRITE_OUTPUT_DIR**: Pass 'True' to enable overwriting the trained model to same directory. (default is 'True')  
**NNCF_CONFIG**: Config file with Neural Network Compression Framework(NNCF) algorithms. (default is '/home/training/config/bert_config.json')  

The above environment variables having default values can also be updated with a different value on the docker run command itself.

#### **Example**:
Default value set to 'DO_EVAL' environment variable is 'True', which can be changed to 'False' by passing the below statement to the docker run command.  

    --env DO_EVAL=False

You can also pass other additional arguments(as per your training script) to the container. This can be done as follows.  

    docker run --env ENV_NAME1=value1 --env ENV_NAME2=value2 image_name --arg3 val3 --arg4 val4
