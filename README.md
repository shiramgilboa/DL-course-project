# Deep Learning – 46211
## Project - DL for tabular data
![image](https://user-images.githubusercontent.com/94564657/215352447-d79c0427-c497-4e0b-863a-7861683c67e8.png)

---

## Project Content
- Dataset - COVID-19 patient's symptoms, status, and medical history
- Data Pre-Processing and Sampling Techniques
- XGBoost and simple FC neural network
- TabNet - deep learning architecture designed by Google to affectively apply deep learning on tabular data
- Additional method for improving results
- Results Comparison

---

## Repository Content
- Two helpers files:
  1. models_helpers.py
  2. pre_proccessing_helpers.py
- Data pre-processing notebook ("data_pre_precossing.ipynb")
- TabNet, XGBoost and FC neural network results notebook ("models.ipynb")
- RTDL results notebook **********
- Raw data - CSV file ("preprocessed_data.csv")
- Pre-processed data ("Covid Data.csv")

---

## How to Use
first install TabNet to your work environment - "pip install pytorch-tabnet" 
Download files into a single folder
for data pre-processing:
  delete '#' from last row in last block
  run pre-processing notebook to create pre-processed data
  
for ready-to-use data:
  use pre-processed data CSV file and run models 

run models.ipynb and ******* to get results

---
## Sources

### TabNet article:
https://arxiv.org/abs/1908.07442 

### Dataset from Kaggle:
https://www.kaggle.com/datasets/meirnizri/covid19-dataset

### TabNet pyTorch documentation:
https://dreamquark-ai.github.io/tabnet/
