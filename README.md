# FedKD_Prox: Towards Enhanced Federated leaning based IoT Intrusion Detection

This repository contains the code and experiments for the paper:
> [Towards Enhanced Federated leaning based IoT Intrusion Detection](https://www.overleaf.com/project/65bbf52583bb92891d46c9c3)
>
> Conference: VTC2024 Fall (https://events.vtsociety.org/vtc2024-fall/)

In the rapidly growing field of Internet of Thing (IoT), there is a surge interest in approaching network security issues through Machine learning (ML) and Deep Learning techniques (DL). However, ML based network intrusion detection (NID) system faces the issue of data privacy and security due to the need for centralized data from the devices. As a distributed learning paradigm, Federated Learning (FL) offers a privacy-preserving solution for training ML model. Many existing works have proven the feasibility of training IoT NID model through FL. However, since most IoT devices are purpose-oriented, data and system heterogeneity across different IoT devices can significantly impact the convergence speed and detection rate of the NID model. Additionally, IoT devices often lack the computing power needed for intensive model training and do not have sufficient communication resources to communicate the model updates with the server. To address these issues, we propose FedKD-Prox, a FL framework based on Federated Proximal(FedProx) and Knowledge Distillation(KD). It improves NID model robustness and reduce computational and communication intensity for IoT devices. Our simulations, using real-world network security dataset, showed that FedKD-Prox improves detection rate and convergence speed while significantly reduces the communication cost for IoT devices.

## General Guidelines
Our codes includes implementation of FedAvg(baseline 1), FedProx(baseline 2), and FedKD-Prox(proposed algoritm).
If you would like to run our codes:
* At least learning rate needs to be fine-tuned based on your metric if you are using different datasets. 
* For baseline FedProx(baseline 2), the author suggests to tune the mu value from [0.001, 0.01, 0.1, 0.5, 1].
* We used a small portion of the original datasets. As a reference, a Nivida RTX 3060 (12GB) GPU requires ~ 8 hours to train 1 million data points for each dataset. You can increase the data volume basd on your GPU capability.
* The program stores the testing accuracy and training loss of the global model on each global round as the training results.

## Preparation
