[![Build Status](https://travis-ci.org/cxz/tgs-tpu.svg?branch=master)](https://travis-ci.org/cxz/tgs-tpu)


# Building
```
$ nvidia-docker build -f ./docker/Dockerfile.gpu -t tf .
```

# Running
```
$ nvidia-docker run -u $(id -u):$(id -g) -v $(PWD):/tpu-dev -it tf
$ nvidia-docker run -it tf
```

# Kubernetes
- https://cloud.google.com/tpu/docs/tutorials/kubernetes-engine-resnet
- https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_k8s.yaml
