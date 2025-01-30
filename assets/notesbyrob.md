# Notes


- Azure procedural goes here
    - Done: Start up an Azure VM, `ssh` there, configure it, clone this repo, create and start a `testenv` environment
        - Note that for Ubuntu the default user is `azureuser`
        - Note that `ssh` connection can be done via VSCode
        - Aside: [Here is the serverless function app tutorial for 544](https://github.com/cloudbank-project/az-serverless-tutorial/blob/main/content/functions/_index.md) and [here is the published version](https://cloudbank-project.github.io/az-serverless-tutorial/)
    - Command line `python -m index.py` with the idea of running CIFAR-10
        - Blocked: GCP credential error
- GCP procedural goes here
- AWS procedural goes here

- Notes from 14-JAN-2025
    - Using Docker is the best way
    - Snapshot the Spot instance? No no no.
    - Create the Docker image in advance on two clouds...
    - Our program looks in the GCP bucket for a checkpoint (CIFAR has 10 iterations)
        - Resumes training based on which checkpoint it found
        - So here is the plan
            - Create an Ubuntu Docker image with Python installed, repo loaded, CIFAR dataset
            - Runs, writes first epoch...
            - GCP instance gets evaporated after this...
                - How does this happen?
                    - Periodic polling: Every 5 seconds must be running on the Pre-VM
                    - Polling code exists already
                    - Docker needs to mount the "manual halt" resource
                    - Demo: touch `preempt.txt`
                    - Now who starts the image on AWS?
            - Start the same same Docker imagine on AWS
                - which is already configured to look at the GCP Bucket; picks up where we left off
- Notes from discussion with Oorjit 18-DEC-2024
    - Three tasks
        - CIFAR-10 is full-blown training, runs 5 minutes, CNN, image classification
        - ImageNet/ResNet: fine-tuning (not full), run time 10 minutes, image classification
            - Basic parameters and timing needed
        - bert sentiment analysis
            - text: fine-tuning, 4 classes, runs in like 5 minutes
    - Two kinds of checkpointing: The hard way and the easy way...
        - Easy way is write a x-kb file upon trigger: works on cifar10 on Google cloud object storage
            - Can simulate the interrupt
            - Don't care so much about alert
        - Hard way is to convert the VM to an image in 30 second
            - Get alert (notification trigger: your machine is about to die): Issue the bundler
            - need a dmtcp (see Hyak) on GCP (or some other cloud)



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


## Today 13-JUN-2024 Items

- Checkpointing
- Preemption-proof execution
    - Test via "Host Maintenance Event" which we can trigger on a GCP Spot instance
    - This is initiated using the `gcloud` cli: See the `README.md`
- CIFAR-10 on "any" cloud (using Google)
- The `index.py` file in ~ works with `~/utils/preemption.py` to do a 5 second polling loop
    - Two threads: One does the work, the other checks for premption
    - Does the work: Training thread: CIFAR-train: See `~/models/cifar.py`
 
### From Scratch CIFAR on GCP SPOT

This is redundant to `~/README.md`.

* Start a SPOT VM on GCP, log in to the command line
* `git clone https://github.com/oorjitchowdhary/ml-training-preemptible-vms.git`
* Create a Google Cloud Storage bucket
    * Can do using console
    * `gcloud storage buckets create gs://BUCKET_NAME --location==BUCKET_LOCATION`
* Establish a service account for use by the Python program
    * Created by default per GCP Project: Dashboard shows this: Login > Project > IAM and Admin
* Make service account usable via service account JSON: Download through this same IAM interface
    * Left menu: Service Accounts > hyperlink `developer.gserviceaccount` etc > Top menu KEYS >
        * Add new Key > JSON > Download this JSON file to local computer
        * `scp` this file to the SPOT instance to `~` where `index.py` resides
        * Yes: At any moment this SPOT could evaporate
        * Incidentally filename will be `<project name>-<numbers>.json`
* Align json creds with the code base
    * go to `~/utils/checkpointing.py` and modify lines 4 and 5
    * `service_account` and `bucket_name` adjusted to match the above steps
* Activate the environment, run the program


```
sudo apt update
cd ml-training-preemptible-vms
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 index.py
```

* Intermediate checkpoint process
    * `index.py` writes to `~/checkpoints/` like `checkpoint_<epoch>.pth` (pytorch file extension)
        * `torch.save(<directory>)`
        * Then the python file copies this to the bucket 
* Timing comparisons
    * First time run of CIFAR will require extra time to download the data; so re-run that
    * Use the Linux `time` utility or Python `time` library to get a runtime
    * Record the results!
    * Stop, re-size, start the VM: Another instance type
* Earlier version of `pip install -r requirements.txt` ran into resolution problems
    * For five packages I deleted versions, as in `==1.2.1`
    * This should not be necessary now
* Re-run:
    * Can first blow away the checkpoint results from earlier runs first.
    * Can first blow away CIFAR source data
    * Check print diagnostics, match to source code: Consistent?
* Automatic deployment: Via VM Image or Container: To Do list
* Sky Pilot pivot
    * Currently we do not use any Sky Pilot functionality
    * SPOT recovery / failover: We wish to continue...
        - SkyPilot needs access to our checkpointing bucket via YAML file
    - Sky Pilot can maintain a cluster
    - Activation of Sky Pilot
        - install `sky` on my laptop
        - clone the cifar repo on my laptop
        - As above: Create the bucket on GCP (although this can also be automated) 
        - add `skypilot.yml` to the repo
            - There actually is one present but it is in "example" mode only
        - `sky launch -c myclustername myymlfilename.yml`
            - usual credentials step
            - then Sky Pilot finds a cheap VM; asks for confirmation on spend etcetera
            - then Sky Pilot creates a checkpoint bucket if none found
            - then Sky Pilot goes on to setup to eventually execute `python3 -m index.py`
            - then Sky Pilot will do a polling loop for preemption
                - So this is happening in parallel to the preemption thread on the SPOT VM
    * During execution Sky Pilot will check `~/checkpoints` for new files: Copy to bucket
        * `torch.save()` creates this file; SkyPilot copies that to the bucket; another polling loop
        * So we now have two redundancies: checkpoint copy to bucket and thread checking for preemption
        * See the sky pilot yaml file reference documentation

## Install and use `gcloud` to simulate a preemption

- Simplest? is to do this on my laptop
- Authenticate `gcloud` by installing it and running it: First time it walks me through authentication
- This involves 'open your browser and authenticate'
- The result is some credentials hidden in a `.gcp` (or something like that) folder in my root folder
- The `gcloud` command to preempt the SPOT instance is in `README.md`



## Legacy table

| Instance        | Rate | Sec  |  Cost | Epochs | Learning Rate | Momentum
| ------------- |:-------------:| -----:| -----:| -----:| -----:| -----:|
| Az B2ms      | .085 | 580 | .014  | 10 | .001  | 0.9 |
| Az D4s v3    | .194 | 452 | .024  | 10 | .001  | 0.9 |
| Az B20ms     | .843 | 760 | .178  | 10 | .001  | 0.9 |
| Az D8ds v5 | .504 | 315 | .044 | 10 | .001 | 0.9 |
