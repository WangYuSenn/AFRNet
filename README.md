# AFRNet
This project provides the code and results for 'Transmission Line Detection Through Auxiliary Feature Registration with Knowledge Distillation'

# Requirements
Python 3.7+, Pytorch 1.5.0+, Cuda 10.2+, TensorboardX 2.1, opencv-python 
If anything goes wrong with the environment, please check requirements.txt for details.

# Architecture and Details
![image](https://github.com/user-attachments/assets/c16c13cd-4072-4cd4-b8e5-7e1a839c6433)
![image](https://github.com/user-attachments/assets/eff902a2-c0dc-4703-99c0-977bfd2ea93b)
![image](https://github.com/user-attachments/assets/00ae6804-0f8e-4e83-9d9b-b1d9cfdb2f7f)

# Results
![image](https://github.com/user-attachments/assets/4c532825-c3e6-470a-a20c-03c95fbd5a1c)
![image](https://github.com/user-attachments/assets/5d9cbaaf-4b53-47fa-9548-4a33b32fe8fd)

# Data Preparation
[数据集链接](https://pan.baidu.com/s/1XrDpcsRAHAXRT4HQik0prA?pwd=YS98)

# Training & Testing
modify the train_root val_root save_path test_path in config.py according to your own data path.

Train the AFRNet:

python train.py

modify the test_path path in test.py according to your own data path.

Test the AFRNet:

python test.py

# Evaluate tools

Implemented through evaluate.py, specific predictions can be obtained by setting the path inside.

# Saliency Maps and Weights

链接: https://pan.baidu.com/s/1lFnIqVR6brI0T2unYOBxUg?pwd=4pfh 提取码: 4pfh

# Pretraining Models

链接: https://pan.baidu.com/s/1AbfpKgbTJPFJFWrxyQAzhw?pwd=j51m 提取码: j51m

# Contact
Please drop me an email for any problems or discussion: https://wujiezhou.github.io/ (wujiezhou@163.com) or wysss1998@163.com


