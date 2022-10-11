 # An End-to-End NLP workflow with Quantization Aware Training using NNCF, and Inference using OpenVINO toolkit/OpenVINO Execution Provider on Azure cloud
## Description
This document contains instructions on how to setup kubernetes and helm for ML workflows.

## System Setup for one Node cluster
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
