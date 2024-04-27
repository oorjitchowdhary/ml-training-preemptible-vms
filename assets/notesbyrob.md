# Notes by Rob


Plan


* Start a VM on Azure
* Follow these steps to run cifar-10 on it
    * Find a tougher job eg from kaggle
* See what can be done about benchmarking it
* Go again on a bigger machine

<BR><BR><BR>

* Clone the repo:

`git clone https://github.com/oorjitchowdhary/cifar-on-spot-vm.git`


* Follow these configuration steps
    * I found I needed to edit `requirements.txt` for `pip install` to run:
        * For five packages: Delete version text, as in `==1.2.1`

```
sudo apt update
cd cifar-on-spot-vm/
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```



| Instance        | Rate | Sec  |  Cost | Epochs | Learning Rate | Momentum
| ------------- |:-------------:| -----:| -----:| -----:| -----:| -----:|
| Az B2ms      | .085 | 580 | .014  | 10 | .001  | 0.9 |
| Az D4s v3    | .194 | 452 | .024  | 10 | .001  | 0.9 |
| Az B20ms     | .843 | 760 | .178  | 10 | .001  | 0.9 |
