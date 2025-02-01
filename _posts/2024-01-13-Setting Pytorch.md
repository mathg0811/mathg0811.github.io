---
title: ML Study note - Setting PyTorch
author: DS Jung
date: 2024-01-13 13:00:00 +0900
categories: [ML, Note, Pytorch]
tags: [studynote]    # TAG names should always be lowercase
comment: true
math: true
mermail: false
pin: true
---

This blog is randomly written without order or flow at the author's convenience or impulse

---
### Setting Environment

Windows 10 ( build 19045 )  
Intel i5-9400F  
NVIDIA GTX 2070 super

--- 
### Installation Objectives PyTorch Environment
1. python 3.9
1. miniconda 3
1. cuda 11.8
1. cudnn 8.9.7 ( latest )
1. pytorch 2.1.2 ( latest )

#### Reason for selecting versions
- cuda 11.8 and 12.1 is recommended for pytorch 2.1
- cuda 12.1 does not support static linkinㅎ

--- 

#### Install Cuda 11.8
link [Cuda 11.8 Download](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)  
Installation Path : C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8

#### install cudnn 8.9.7 (latest)
link [Cudnn lastest Download](https://developer.nvidia.com/rdp/cudnn-download)

#### Install miniconda
link [miniconda Download](https://docs.conda.io/projects/miniconda/en/latest/index.html_)

---
#### Reason for selecting miniconda
- If you use pip when creating a Python virtual environment, this environment may be path dependent and unavailable to implement when using IDE such as VScode does not support custom work path
- Anaconda is heavy because it installs too many packages together from the start

---
#### Conda create virtual environment


