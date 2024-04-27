![UW eScience Institute](https://www.washington.edu/research/wp-content/uploads/eScience_Logo_HR.png)

# ML training on Spot VMs
This repository represents an ongoing cloud computing effort at the [UW eScience Institute](https://escience.washington.edu) that aims to optimize cloud usage for research computing. The overarching goal is to find the most prevalent cloud use cases for research and approximate a walkthrough alongside a breakdown of the cost analysis and performance metrics for each use case.

## Overview
Currently, the repository focuses on training a convolutional neural network (CNN) machine learning model on Spot VM instances on GCP. Spot VM instances, unlike regular VM instances, are preemptible and can be terminated by Google at any time. This makes them significantly cheaper than regular VM instances, but also less reliable. We aim to design a workflow that can take advantage of the cost savings of Spot VMs while minimizing the impact of potential interruptions.

### How?
Through periodic checkpointing of the model in training to a cloud storage bucket, we can iteratively save the model's progress and resume training from the last checkpoint in case of a VM interruption.

## Technical Details
**Dataset:** CIFAR-10, a popular dataset for image classification tasks. The dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

**Model:** A classic CNN model with 2 convolutional layers, a max-pooling layer, and 3 fully connected layers. The model is trained using the Adam optimizer and the categorical cross-entropy loss function.

**Cloud Storage:** A Google Cloud Storage bucket stores the `.pth` checkpoint files of the model.

### :construction: WIP: A note on SkyPilot
There is a `skypilot.yaml` file initialized in the repository that contains the configuration for utilizing [SkyPilot](https://skypilot.readthedocs.io/en/latest/examples/spot-jobs.html), a framework for running jobs across clouds. A successful and complete implementation of the workflow would involve using SkyPilot to automatically find available Spot VM resources across multiple cloud providers and restart the job from the last checkpoint in case of a preemption.

## Getting Started
To see the workflow in action, follow the installation instructions below:
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