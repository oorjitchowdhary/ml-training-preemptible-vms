# Notes by Rob

## ImageNet

- 155GB, 1000 image classifications; 1.2M images, 75 to 90 epochs
- Data is on kaggle: Would need to join, download, upload to S3
    - Is there another path to this data? Unknown
- Need more than 3 layers of neurons
- ResNet is a modification of the CNN: 50 layers all interlinked (MSR 2015 paper)
    - Deep learning model: 2 variants
        - Resnet 18: Depth is 18 layers
        - Resnet 50: Depth is 50: One of best approaches to ImageNet
            - PyTorch already has PyTorch.resnet50(usePretrained = True) and go from there
    - There is also AlexNet and VGG not considered here
- ImageNet: The right thing to use?
    - SkyPilot's discovered S3 bucket of ImageNet data is list-only (no read)
    - Other datasets that will require a GPU: Open question at this point (June 2024)
        - Perhaps CIFAR-100 (more images)
        - Perhaps MNIST (handwritten digits: 28 x 28 60k images)

## Plan

* Start a VM on Azure
* Clone the repo:
    * `git clone https://github.com/oorjitchowdhary/cifar-on-spot-vm.git`
* Follow the configuration process
    * Configure a storage bucket for checkpoints
    * Establish a service account for use by the Python program
* `while True:`
    * Activate the environment, run the program
        * These commands are out of date; just as a reminder:
            * `source .venv/bin/activate`
            * `python -m index.py`
        * First time run will require extra time to download the data; so re-run that
    * Use the Linux `time` utility or Python `time` library to get a runtime
    * Record the results
    * Stop, re-size, start the VM: Another instance type

<BR><BR><BR>



## Environment prep

* `pip install -r requirements.txt` ran into resolution problems
    * For five packages I deleted versions, as in `==1.2.1`
  

Here is the old command sequence:

```
sudo apt update
cd ml-training-preemptible-vms/
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m index.py
```

Accuracy around 63% after 10 epochs.


| Instance        | Rate | Sec  |  Cost | Epochs | Learning Rate | Momentum
| ------------- |:-------------:| -----:| -----:| -----:| -----:| -----:|
| Az B2ms      | .085 | 580 | .014  | 10 | .001  | 0.9 |
| Az D4s v3    | .194 | 452 | .024  | 10 | .001  | 0.9 |
| Az B20ms     | .843 | 760 | .178  | 10 | .001  | 0.9 |
| Az D8ds v5 | .504 | 315 | .044 | 10 | .001 | 0.9 |
