# recipes

The first goal of this repository is to run a small ML task within a Docker container
on a cloud VM that is *not* a preemptible instance. We will *simulate* a preemption
using a signal (Linux: file `touch`) so that the Docker container notices it is about 
to be shut down. It shuts down gracefully and we can re-start that job from where 
it left off. 


The second goal: Once the above machinery is established we will try it on the spot 
market.


The third goal: Run on three different cloud spot markets: AWS, GCP and Azure.


This document narrates the construction process for an intended audience of "new 
to cloud computing / new to containerization". 


## Containers


We suggest that containerizing workflows is a good option to know about in research computing. 
*Containers* have merit beyond our stated context, which is stretching one's cloud computing
budget by a factor of 3 or more by using preemptible instances. 


We begin with the container "story arc" in three stages, from file to image to container. 
This repository contains a short text file called a *Docker file*. It consists of a terse 
set of instructions. These instructions are used to construct a *Docker image*. This is a
file typically several hundred Megabytes in size. A Docker image is typically uploaded to
some web-accessible location such as DockerHub. When stored there it is inert; it does not
do anything. However a Docker image can be downloaded to a Docker-enabled virtual machine
where it runs in a Docker *container*. 


The Docker container is connected to its external or host VM environment as well as to 
the internet; so a running Docker container can notice if an interruption is imminent; 
and it can take action to shut down gracefully. 


The Docker progression from *file* to *image* to *container* is managed through a
utility program appropriately named `docker`.  As a first step we 
ensure `docker` executes by installing it in a development VM. Once
installed we can issue `docker` commands such as `docker build` and `docker run`.


## Libraries


The machine learning task at hand is to train a Convolutional Neural Network (CNN) 
using the Python deep learning library `pytorch`. The `pytorch` library and
other related libraries must be installed through an automated mechanism. 
This is most commonly done using a package manager such as `pip` or `conda`.
In this instance we use `pip`. 


