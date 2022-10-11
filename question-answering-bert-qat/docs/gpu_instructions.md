# Description
This document describes how the helm chart can be modified to enable training pod on a NVIDIA GPU

One the GPU node, please run the below commands

1. Deploy NVIDIA GPU device plugin on the GPU node to be able to access gpu through pods.

```
   sudo wget https://k3d.io/v4.4.8/usage/guides/cuda/config.toml.tmpl -O /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl
   sudo systemctl restart k3s
   kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.12.3/nvidia-device-plugin.yml
```

2. Replace the below code to pre_install_job.yaml from line 31.
```
     containers:
        - name: w4-training
          image: "{{ .Values.image.registry }}/{{ .Values.image.trainingimage }}"
          imagePullPolicy: "IfNotPresent"
          command: ["/bin/sh"]
          args: ["-c", "cd /home/training && ls -l && ./run_qat.sh"]
          resources:
            limits:
              nvidia.com/gpu: 1
          env:
            {{- range $key, $val := .Values.trainingenv }}
             - name: {{ $key }}
               value: {{ $val | quote}}
            {{- end }}
             - name: CUDA_HOME
               value: /usr/local/cuda
          volumeMounts:
            - name: training
              mountPath: /home/training
            - name: cudapath
              mountPath: /usr/local/cuda

      volumes:
        - name: training
          hostPath:
             path: {{.Values.mountpath.trainingvolume}}
        - name: cudapath
          hostPath:
             path: /usr/local/cuda
```


## Troubleshooting

Sometimes we encounter

```
 RuntimeError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 15.78 GiB total capacity; 14.29 GiB already allocated; 3.
 12 MiB free; 14.31 GiB reserved in total by PyTorch)
```

Please reduce the PER_DEVICE_TRAIN_BATCH_SIZE and restart the training. 