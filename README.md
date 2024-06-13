![UW eScience Institute](./assets/logo.jpg)

# ML training on preemptible VMs
This repository represents an ongoing cloud computing effort at the [UW eScience Institute](https://escience.washington.edu) that aims to optimize cloud usage for research computing. The overarching goal is to find the most prevalent cloud use cases for research and approximate a walkthrough alongside a breakdown of the cost analysis and performance metrics for each use case.

## Overview
At the moment, the repository showcases training a basic convolutional neural network (CNN) model on the CIFAR-10 dataset as well as fine-tuning a pre-trained ResNet50 model on the ImageNet dataset. The training is done on Google Cloud Platform (GCP) using its Spot VM instances, which are significantly cheaper than regular VM instances but are preemptible and can be terminated by Google at any time. We aim to design a workflow that can take advantage of the cost savings of Spot VMs while minimizing the impact of potential interruptions.

### How?
Through periodic checkpointing of the model in training to an external cloud storage bucket, we can iteratively save the model's progress and resume training from the last checkpoint in case of a VM interruption.

## Technical Details

### CIFAR-10 task
**Dataset:** CIFAR-10, a popular dataset for image classification tasks. The dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

**Model:** A classic CNN model with 2 convolutional layers, a max-pooling layer, and 3 fully connected layers. The model is trained using the Adam optimizer and the categorical cross-entropy loss function.

**Cloud Storage:** A Google Cloud Storage bucket stores the `.pth` files of the model.

### ImageNet task
**Dataset:** ImageNet subset, a subset of the ImageNet dataset. The subset dataset consists of 1.28 million images spanning 1000 classes.

**Model:** A pre-trained ResNet50 model is fine-tuned on the ImageNet subset dataset. The model is trained using the Adam optimizer and the categorical cross-entropy loss function.

**Cloud Storage:** A Google Cloud Storage bucket stores the `.pth` files of the model.

## Workflow
### Training

**CIFAR-10:**
1. Load and normalize CIFAR-10 using `torchvision`. (The output of `torchvision` datasets are of type `PILImage` images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1].)
2. Define the 3 layer CNN model.
3. Define the loss function and optimizer.
4. Loop over the dataset `n` times, feeding inputs to the model and optimizing the weights.

Relevant tutorial: [Training a Classifier in PyTorch](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

**ImageNet:**
1. Load the pre-trained ResNet50 model from `torchvision.models`.
2. Replace the final fully connected layer with a new one that has the same number of outputs as the number of classes in the dataset.
3. Define the loss function and optimizer.
4. Loop over the dataset `n` times, feeding inputs to the model and optimizing the weights.

Relevant tutorial: [ImageNet training in PyTorch](https://github.com/pytorch/examples/tree/main/imagenet)

### Checkpointing
1. Periodically save (at the end of each epoch) the model's state dictionary to a `.pth` file in `/checkpoints`.
2. Upload the `.pth` file to a cloud storage bucket.


**Cloud Storage bucket:**<br>
A Google Cloud Storage bucket is used to store the `.pth` files of the model. The bucket is created using the `google-cloud-storage` Python package. Setting up such a bucket involves creating or using the default service account, downloading the JSON key file, and setting the `service_account_json` variable in `checkpointing.py` to the path of the JSON key file. Make sure to also replace the `bucket_name` variable with the name of your bucket.

Note: You can use any cloud storage service to store the model checkpoints as long as you modify the `checkpointing.py` script accordingly.

### Preemption Handling
Detecting a preemption event involves different strategies depending on the cloud provider. However, a popular approach is to poll the metadata server for a preemption event.

In this repository, we demonstrate concurrently polling the metadata server for a preemption event every 5 seconds while training a model. This approach uses Python's `threading` module to run the polling function in a separate thread. If a preemption event is detected, the model's state is saved and uploaded to the assigned cloud storage bucket, making the script exit gracefully.

**Simulating a preemption event:**<br>
On GCP, you can simulate a preemption via a host maintenance event. Read more [here](https://cloud.google.com/compute/docs/instances/simulating-host-maintenance).

```bash
gcloud compute instances simulate-maintenance-event VM_NAME --zone ZONE
```

Relevant resources for other cloud providers:
- [Azure interruptible workloads](https://learn.microsoft.com/en-us/azure/architecture/guide/spot/spot-eviction)
- [Getting started with AWS EC2 Spot](https://aws.amazon.com/ec2/spot/getting-started/)

## A note on SkyPilot
There is a `skypilot.yaml` file initialized in the repository that contains the configuration for utilizing [SkyPilot](https://skypilot.readthedocs.io/en/latest/docs/index.html), a framework for running jobs across clouds.

TODO: Notes about task.yaml and auto-failover. 

## Getting Started
1. If you haven't already, [create a Google Cloud Storage bucket](https://cloud.google.com/storage/docs/creating-buckets) and replace the `bucket_name` variable in `checkpointing.py` with the name of your bucket.
2. [Create a service account](https://cloud.google.com/iam/docs/creating-managing-service-accounts); download the JSON key file and replace the `service_account_json` variable in `checkpointing.py` with the path to the JSON key file.
3. To see the workflow in action, follow the installation instructions below:
```bash
# Clone the repository
git clone https://github.com/oorjitchowdhary/cifar-on-spot-vm.git
cd cifar-on-spot-vm

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt

# Run the script
python index.py
```