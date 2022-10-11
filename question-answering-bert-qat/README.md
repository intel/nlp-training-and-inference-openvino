# An End-to-End NLP workflow with Quantization Aware Training(QAT) using Neural Networks Compression Framework (NNCF), and Inference using OpenVINO™ & OpenVINO™ Execution Provider
This document details instructions on how to run quantization aware training & inference pipeline with Helm and docker.
 *	Introduction
    *	[Description](#Description)
    *	[Block Diagram](#block-diagram)
    *	[Project Structure](#project-structure)
    * [Prerequisites](#prerequisites)
      *	[Hardware](#hardware)
      * [Software](#software)
      *	[Cloning the repository](#download-source-code)
      *	[Docker Images](#docker-images)      
  *	Get Started
    *	[QAT – Parameters to be modified](#modify-helmchartqatvaluesyaml)
    *	Options to execute the workflow
        *	Using Helm 
            *	[Steps to use the Helm chart](#helm-usage)
            *	[Use Case 1: Quantization Aware Training with Inference using OpenVINO™ Integration with Optimum*](#usecase-1)
            *	[Use Case 2: Quantization Aware Training with Inference using OpenVINO™ model server](#usecase-2)
            *	[Use Case 3: Quantization Aware Training with Inference using OpenVINO™ Execution Provider](#usecase-3)
            *	[Use Case 4: Only Inference](#usecase-4)
            *	[Clean up](#clean-up)
            * [Output](#output)
        *	Local Build instructions using Docker run
            *	[Quantization Aware Training](https://github.com/intel/nlp-training-and-inference-openvino/tree/main/question-answering-bert-qat/quantization_aware_training/README.md) 
            *	[Inference using OpenVINO™ Integration with Optimum*](https://github.com/intel/nlp-training-and-inference-openvino/tree/main/question-answering-bert-qat/openvino_optimum_inference/README.md)
            * [Inference using OpenVINO™ Model Server](https://github.com/intel/nlp-training-and-inference-openvino/tree/main/question-answering-bert-qat/openvino_optimum_inference/README.md)
            *	[Inference using OpenVINO™ Execution Provider](https://github.com/intel/nlp-training-and-inference-openvino/tree/main/question-answering-bert-qat/onnxruntime_inference/README.md)
            *	[Clean up](https://docs.docker.com/engine/reference/commandline/rm/)
    * Optional: 
        * [Set Up Azure Storage](#set-up-azure-storage)
        * [Enable NVIDIA GPU for training](https://github.com/intel/nlp-training-and-inference-openvino/tree/main/question-answering-bert-qat/docs/gpu_instructions.md)
    * [References](#references)
    *	[Troubleshooting](#Troubleshooting)
    
## Description
The main objective of this automated AI/ML pipeline is to demonstrate quantization aware training using Neural Networks Compression Framework [NNCF] through OpenVINO™ Integration with Optimum* and deployment of inference application through various APIs i.e., Hugging Face API, Onnxruntime API and OpenVINO™ Model Server

## Block Diagram

![AI workflow](https://github.com/intel/nlp-training-and-inference-openvino/tree/main/question-answering-bert-qat/docs/usecase_flow.png)

This workflow is stitched together and deployed through Helm by using microservices/docker images which can be built or taken from Azure Marketplace.

The workflow executes as follows
1) The Pipeline triggers Quantization Aware Training of an NLP model from Hugging Face. The output of this container is the INT8 optimized model stored on a local/cloud storage.
2) Once the model is generated, then inference applications can be deployed with one of the following APIs  
    i) Inference using ONNXRT API  
   ii) Inference using Hugging Face API  
  iii) Deploy the model using OpenVINO™ Model Server and send in grpc requests  

## Project Structure 
```
├──quantization_aware_training - Training related scripts
├──openvino_optimum_inference - Hugging Face API inference scripts with OpenVINO™ Runtime
├──onnxruntime_inference - ONNX Runtime API inference scripts with OpenVINO™ Runtime
├── helmchart
 ├── deployment_yaml
   ├──deployment_onnx.yaml - Deploys onnxruntime with OpenVINO™ container
   ├──deployment_ovms.yaml - Deploys optimized model through OpenVINO™ Model Server
 ├── qat
  ├── charts
  ├── templates
   ├──pre_install_job.yaml - Deploys Quantization Aware Training container
   ├──deployment_openvino_optimum.yaml - Deploys Huggingface API inference container 
 ├── chart.yaml
 ├── values.yaml
 └── README.md
├── Dockerfiles
 ├──Dockerfile_openvino_optimum - Docker file to build OpenVINO™ Integration with Optimum[NNCF] image
 ├──docker-compose.yaml - Docker compose to build docker image
 ├──requirements_openvino_optimum - requirements file to be installed for the OpenVINO™ Integration with Optimum
```
## Prerequisites 

- Kubernetes cluster that has comparable configurations as follows

## Hardware
### Edge Nodes

-   One of the following System:

    -   Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz (16 vCPUs)
.
    -   Intel(R) Xeon(R) with NVIDIA* GPU.

-   At least 64 GB RAM.

-   At least 256 GB hard drive.

-   An Internet connection.

-   Ubuntu\* 20.04 LTS
    
## Software

- Docker & Docker Compose installation [here](https://docs.docker.com/engine/install/ubuntu/)
   
- Any flavor of kuberentes variations.

  We have used rancher k3s installation. Further details in [here](https://rancher.com/docs/k3s/latest/en/installation/install-options/#options-for-installation-with-script)
  ```
  curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 644
  export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
  ```
- Helm installation on master node. 

  Steps to install as below. For further details please refer [here](https://helm.sh/docs/intro/install/)
   ```
   curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
   chmod 700 get_helm.sh
   ./get_helm.sh
   ```
- We are currently using  `bert-large-uncased-whole-word-masking-finetuned-squad`
 model for `Question Answering` usecase through quantization aware training and inference . We do have training and inference scripts in the respective folders.

## Install and Run the workflow:
## Download source code

```
git clone  https://github.com/intel-innersource/frameworks.ai.edgecsp.quantization-training-inference.git

cd frameworks.ai.edgecsp.quantization-training-inference
```
## Docker images 
Build docker images to store in local image registry or pull image from Azure Marketplace. Please follow either Option i or Option ii 

   - Option i. Steps to build images locally.

      Start a local registry (you can use any string for `<registry_name>` e.g. qat_docker_registry)
       
       ```
       docker run -d -p 5000:5000 --restart=always --name <registry_name>  registry:2
       ```

      Build docker image. Please edit the `docker-compose.yaml` file for `<registry>` tag with the local or private registry address. If using local registry, edit it to "localhost:5000"
         
      ```
      cd dockerfiles
      docker compose build
      ```
	  
      Push the image to the local image registry
         
      ```
      docker compose push openvino_optimum  
      cd ..  
      ```

   - Option ii. Images to be pulled from Azure Marketplace
       
        Subscribe to the images mentioned below  
        To be updated with screenshots once we know the name and place of the images  
        Subscribed images from Azure Marketplace will get stored onto your private registry
        
        Create a kubectl secret for the private registry where the images are stored onto.

        ```
        kubectl create secret docker-registry <secret_name>
            --docker-server <registry_name>.azurecr.io
            --docker-email <your_email>
            --docker-username=<service_principal_id>
            --docker-password <your_password>
        ```
	
        Edit the `helmchart/qat/values.yaml` with the secret name for the imagePullSecrets field with the <secret_name> as above.

        Edit the `helmchart/qat/values.yaml` with the <registry_name> for repo_name field

## Modify helmchart/qat/values.yaml
  * Replace <registry_name> under 'image:' with the user's private registry path or with the 'localhost:5000' for the local registry name.  
  * Replace <current_working_gitfolder> under 'mountpath:' with the current working repo directory. 
### **Note:**  
  Relative paths do not work with Helm.
  * Edit the `helmchart/qat/values.yaml` file for the <train_node> and <inference_node> values under 'nodeselector' key  
    Pick any of the available nodes for training and inference with the nodename of this command.
    ``` 
      kubectl get nodes --show-labels
    ```
    nodeselector:  
      trainingnode: <train_node>  
      inferencenode: <inference_node>

   
  * Edit `helmchart/qat/values.yaml` file with higher values of MAX_TRAIN_SAMPLES and MAX_EVAL_SAMPLES parameters for better finetuning of data. Default value is 50 samples.
  * More details on all the parameters in [here](https://github.com/intel/nlp-training-and-inference-openvino/tree/main/question-answering-bert-qat/docs/params_table.md)
   
## Helm Usage
 Steps to install Helm chart with both training and inference . More details on Helm commands [here](https://helm.sh/docs/helm/helm_install/)
   
 
### Usecase 1:

  QAT with Inference using OpenVINO™ Integration with Optimum.

   Training pod is deployed through `pre_install_job.yaml`.
   Inference pod is deployed through `deployment_optimum.yaml`.

  
   ```
    cd helmchart
    helm install qatchart qat  --timeout <time>
   ```
  `time` value. For the above H/W configuration and with MAX_TRAINING_SAMPLES=50 it would ideally be 480**s**. Can increase the value for reduced h/w configuration Please refer to [Troubleshooting](#Troubleshooting) in case of timeout errors.

  Confirm if the training has been deployed.

   ```
   kubectl get pods
   ```
  
  If the training pod is in "Running" state then the pod has been deployed. Otherwise, check for any errors by running the command 
  
  ```
  kubectl describe pod <pod_name>
  ```
  
  The training pod will be in "Completed" state after it finishes training. Refer to [output](#training-output)
  
  Once the training is completed, inference pod gets deployed automatically. Inference pod uses OpenVINO™ Runtime as backend to Hugging Face APIs and takes in model generated from training pod as input.
   
#### Optimum Inference output

 1. Output of the OpenVINO™ Integration with Optimum* inference pod will be stored in the openvino_optimum_inference/logs.txt file. 

 2. You can view the logs using 
    ```
    kubectl logs <pod_name>
    ```
### Usecase 2:
  QAT with Inference using OpenVINO™ Model Server
   
   Training pod is deployed through `pre_install_job.yaml`.
   OpenVINO™ Model Server pod is deployed through `deployment_ovms.yaml`.

   Copy `deployment_ovms.yaml` from `helmchart/deployment_yaml` folder into `helmchart/qat/templates`. Make sure there is only one deployment_*.yaml file in the templates folder for single deployment. 
   
   Follow same instructions as [Usecase1](#usecase-1)
  
  #### OpenVINO™ Model Server Inference output
  1. OpenVINO™ Model Server deploys optimized model from training container.You can view the logs using 
  ```
  kubectl logs <pod_name>
  ```
  2. The client can send in grpc request to server through Hugging Face API user application using the openvino_optimum image .
   For more details on the OpenVINO™ Model Server Adapter API [here](https://docs.openvino.ai/latest/omz_model_api_ovms_adapter.html) Find the ip address of the system where the OpenVINO™ Model Server has been deployed.
  3. Change the 'registry' tag  & 'hostname' in the commnad below before running
 
 'hostname' : hostname of the node where the OpenVINO™ Model Server has been deployed.  
 'registry' : It should be local or private registry address. If using local registry, edit it to "localhost:5000".
 ```
   kubectl get nodes  
     
   azureuser@SRDev:~/frameworks.ai.edgecsp.quantization-training-inference-master/openvino_optimum_inference$ kubectl get nodes  
   NAME    STATUS   ROLES                  AGE   VERSION  
   srdev   Ready    control-plane,master   16d   v1.24.6+k3s1   
     
   In this case, hostname should be srdev
```

```
   cd <gitrepofolder>/openvino_optimum_inference
   docker run -it --entrypoint /bin/bash --env MODEL_NAME=bert-large-uncased-whole-word-masking-finetuned-squad --env MODEL_PATH=<hostname>:9000/models/bert --env MODEL_TYPE=ov  --env ADAPTER=ovms --env ITERATIONS=100 --env INFERENCE_SCRIPT=/home/inference/inference_scripts/bert_qa.py -v  $(pwd):/home/inference/ <registry>/openvino_optimum -c "/home/inference/run_openvino_optimum_inference.sh"
```

  ### Usecase 3:
  QAT with Inference using OpenVINO™ Execution Provider
  
   Training pod is deployed through `pre_install_job.yaml`.
   ONNX Runtime with OpenVINO™ Execution Provider pod is deployed through `deployment_onnx.yaml`. 

   Copy `deployment_onnx.yaml` from `helmchart/deployment_yaml` folder into `helmchart/qat/templates`. Make sure there is only one deployment_*.yaml file in the templates folder.
   
   Follow same instructions as [Usecase1](#usecase-1)
   
#### Onnxruntime Inference output
 1. Output of the onnxruntime inference pod will be stored in the onnxruntime_inference/logs.txt file. 

 2. You can view the logs using 
    ```
    kubectl logs <pod_name>
    ```

### Usecase 4:   
 Only Inference, make sure you have access to the model file and also edit the model path in the `helmchart/qat/values.yaml` file. If the user wants to deploy only inference and skip the training please use command below

```
cd helmchart
helm install qatchart qat --no-hooks
```

    
 #### Clean Up   
  To clean or uninstall helmchart
   ```
   helm uninstall qatchart
   ```
### Output

To view the pods that are deployed. 

   ```
   kubectl get pods
   ```
Take the pod_name from the list of pods
    
    
   ```
   kubectl logs <pod_name>
   ```
    
 If the pods are in completed state it means they have completed the running task. 
 
### Training output

 1. Output of the training container will be an optimized INT8 model generated in the quantization_aware_training/model folder.
 2. Verify if all the model files are generated in the <output> folder.
 3. logs.txt file is generated to store the logs of the training container which will have accuracy details

 

## Optional or Additional Steps:
### Steps to skip training and deploy inference applications:
Before triggering the inference, make sure you have access to the model file and also edit the model path in the `qat/values.yaml` file
   ```
   helm install qatchart qat --no-hooks
   ``` 
	
Cleanup resources: 
	
   ```	  
   helm uninstall qatchart
   ```
### Steps to trigger just one inference application:
Before triggering the inference, make sure you have access to the model file and also edit the model path in the `qat/values.yaml` file

Keep only one deployment-*.yaml file in the qat/templates folder to deploy just one inference application.

1) For Onnxruntime with OpenVINO-EP, use `deployment_onnx.yaml` file. Model format acceptable is .onnx
2) For Huggingface API  with OpenVINO™ runtime, use `deployment_optimum.yaml`. Model format acceptable is pytorch or IR.xml
3) For OpenVINO™ model server, use `deployment-ovms.yaml`. Model format acceptabe is IR.xml


   ```
   helm install qatchart qat --no-hooks
   ``` 
Cleanup resources: 
	
   ```	  
   helm uninstall qatchart
   ```
## Set Up Azure Storage

This is an optional step. Use Azure Storage for multi node kubernetes setup if you want to use the same storage across all the nodes.

### Azure References
  * [Azure File Storage](https://docs.microsoft.com/en-us/previous-versions/azure/virtual-machines/linux/mount-azure-file-storage-on-linux-using-smb)
### Setup Steps
  * Open Azure CLI terminal on Azure Portal 
  * Create a resource group
  ```
    az group create --name myResourceGroup --location eastus
  ```
  * Create Storage Account
  ```
    STORAGEACCT=$(az storage account create \
    --resource-group "myResourceGroup" \
    --name "mystorageacct$RANDOM" \
    --location eastus \
    --sku Standard_LRS \
    --query "name" | tr -d '"')
  ```
  * Create Storage Key
  ```
    STORAGEKEY=$(az storage account keys list \
    --resource-group "myResourceGroup" \
    --account-name $STORAGEACCT \
    --query "[0].value" | tr -d '"')
  ```
  * Create a file share
  ```
    az storage share create --name myshare \
    --quota 10 \
    --account-name $STORAGEACCT \
    --account-key $STORAGEKEY
  ```
  * Create a mount point
  ```
     mkdir -p /mnt/MyAzureFileShare
  ```
  * Mount the share
  ```
     sudo mount -t cifs //$STORAGEACCT.file.core.windows.net/myshare /mnt/MyAzureFileShare -o vers=3.0,username=$STORAGEACCT,password=$STORAGEKEY,serverino
  ```
  ### Usage of Azure Storage in helm
  * Clone the git_repo in /mnt/MyAzureFileShare and make it as your working directory
  * Edit <current_working_directory> in `./helmchart/qat/values.yaml` file to reflect the same
  * All other instructions will be same as in above steps to install the Helm chart and trigger the pipeline. 
  * Once the training is completed, you can view the Azure Portal and check in your fileshare that the model has been generated.

## References
* [OpenVINO™ Integration with Hugging Face Optimum](https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/optimum)
* [NNCF](https://github.com/openvinotoolkit/nncf)
* [Huggingface Transformers training pipelines](https://github.com/huggingface/transformers/tree/main/examples/pytorch)
* [OpenVINO™ Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)

## Troubleshooting

### Connection refused
If encounters connection refused as in below

```
  Error: INSTALLATION FAILED: Kubernetes cluster unreachable: Get "http://localhost:8080/version": dial tcp 127.0.0.1:8080: connect: connection refused
```
Please set the environment variable
```
   export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
```

### Helm Installation Failed

 ```
   Error: INSTALLATION FAILED: cannot re-use a name that is still in use
 ```
Please do
```
	   helm uninstall qatchart
``` 
And then install it again
```
	   helm uninstall qatchart
```
### Helm timeout
 One issue we are seeing currently is if the training is taking longer time we see a timeout error during helm install command
   ```
	Error: INSTALLATION FAILED: failed pre-install: timed out waiting for the condition
   ```
	
 #### Workaround 1
  Based on the system performance please add --timeout <seconds> to the helm command
  ```
    helm install qatchart qat --timeout <time>
  ```
  `time` value. For the above H/W configuration and with MAX_TRAINING_SAMPLES as 50 it would ideally be 480s. Please increase the timeout if need to finetune on whole dataset.
  
 #### Workaround 2
  1. Even if the helm issues error the training pod will get schedule and will keep running and finish its job. Please verify
	kubectl logs <training_pod>
      Once, the pod is completed .
  2. Please do
	```
	   helm uninstall qatchart
	```
  3. Now install the qatchart with just inference as training has completed
	```
	helm install qatchart qat --no-hooks
	```
 
### Useful commands:

Uninstalling helm: (If required)
```
sudo rm -rf /usr/local/bin/helm
```

Uninstalling k3s: (If required)
```
/usr/local/bin/k3s-uninstall.sh
```
[Read more here](https://rancher.com/docs/k3s/latest/en/installation/uninstall/#:~:text=If%20you%20installed%20K3s%20using,installation%20script%20with%20different%20flags) <br />

	

