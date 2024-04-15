# Notes by Rob


Plan


* Start a VM on Azure
* Follow these steps to run cifar-10 on it
    * Find a tougher job eg from kaggle
* See what can be done about benchmarking it
* Go again on a bigger machine

<BR><BR><BR>

```
sudo apt update
python3
git clone https://github.com/oorjitchowdhary/cifar-on-spot-vm.git
cd cifar-on-spot-vm/
sudo apt install python3.8-venv
python3 -m pip install --upgrade pip
python3 -m pip --version
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Last command fails ERROR could not resolve some package; contourpy maybe.
