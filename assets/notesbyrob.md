# Notes by Rob

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
cd cifar-on-spot-vm/
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
