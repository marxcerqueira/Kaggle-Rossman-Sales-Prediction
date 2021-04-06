# Rossmann Sales Prediction

<p align="center"><<img src="https://media-cdn.tripadvisor.com/media/photo-s/16/e1/ec/f7/rossmann.jpg" align="center" ALT="HTML" width="alt%"/></p>

## Introduction

An end-to-end Data Science project with a regression adapted for time series as solution was created four machine learning models to forecast the sales. Predictions can be accessed by users through a bot from the smartphone app Telegram.

This repository contains the solution for a Kaggle competition problem: https://www.kaggle.com/c/rossmann-store-sales

This project is part of the "Data Science Community" (Comunidade DS), a study environment to promote, learn, discuss and execute Data Science projects. For more information, please visit (in portuguese): https://sejaumdatascientist.com/
The goal of this Readme is to show the context of the problem, the steps taken to solve it, the main insights and the overall performance.

**Project Development Method**

The project was developed based on the CRISP-DS (Cross-Industry Standard Process - Data Science, a.k.a. CRISP-DM) project management method, with the following steps:

- Business Understanding
- Data Collection
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Data Preparation
- Machine Learning Modelling and fine-tuning
- Model and Business performance evaluation / Results

&nbsp; 
  <p align="center">
    <img width="50%" alt="drawing" src="https://miro.medium.com/max/700/1*JYbymHifAk7aQ1pHm_IdMQ.png">
  </p>
  &nbsp; 

#### This project was made by Marx Cerqueira.

---

## Table of Contents
- [Introduction](#introduction)
- [1. Business Problem](#1-business-problem)
- [2. Solution Strategy](#2-solution-strategy)
- [3. Top 3 Data Insights](#3-top-3-data-insights)
- [4. Machine Learning Model Applied](#4-machine-learning-model-applied)
- [5. Machine Learning Model Performance](#5-machine-learning-model-performance)
- [6. Business Results](#7-bussines-results)
- [7. Conclusion](#8-conclusion)
- [8. Lessons Learned](#9-lessons-learned)
- [9. Next Steps to Improve](#10-next-steps-to-improve)
- [10.References](#references)

---

# 1. Business Problem.

**The Rossmann Sales Company**

- A private drug store chain based in Germany, with main operations on Europe. Operates over 3,000 drug stores in 7 different contries.
- Offers heathcare and beauty product, including baby and body care, hygiene, cosmetics, dental hygiene, hair care, and so on.
- Business Model: Product sales.

**Problem**

- The CFO wanted to reinvest in all stores, therefore, he need to know how much revenue each store will bring so he can invest it now.

**Goal**

- Predict the daily sales of all stores for up to six weeks in advance.

**Deliverables**

- Model's performance and results report with the following topics:
    - What's the daily sales in dollars for the next 6 weeks?
    - Predictions will be available through a Telegram Bot where stakeholders can acess the prediction by a smartphone
   
[back to top](#table-of-contents)

# 2. Solution Strategy

The strategy adopted was the following:

**Step 01. Data Description:**  I searched for NAs, checked data types (and adapted some of them for analysis) and presented a statistical description.

**Step 02. Feature Engineering:** New features were created to make possible a more thorough analysis.

**Step 03. Data Filtering:**  Entries containing no information or containing information which does not match the scope of the project were filtered out.

**Step 04. Exploratory Data Analysis:**  I performed univariate, bivariate and multivariate data analysis, obtaining statistical properties of each of them, correlations and testing hypothesis (the most important of them are detailed in the following section).

**Step 05. Data Preparation:** This step is necessary both for feature selection and for the machine learning models. Regarding the data types, numerical data was rescaled and categorical data was encoded.

**Step 06. Feature Selection:** The statistically most relevant features were selected using the Boruta package. 
In the next steps, the machine learning models trained by using the features selected by Boruta presented a better generalizability performance.

**Step 07. Machine Learning Modelling:** Some machine learning models were trained. The one that presented best results after cross-validation went through a further stage of hyperparameter fine tunning to optimize the model's generalizability.

**Step 08. Hyperparameter Fine Tunning:**

**Step 09. Convert Model Performance to Business Values:**

**Step 10. Deploy Model to Production:** The model is deployed on a cloud environment to make possible that other stakeholders and services access its results.

# 3. Top 3 Data Insights

**Hypothesis 01:**

**True/False.**

**Hypothesis 02:**

**True/False.**

**Hypothesis 03:**

**True/False.**

# 4. Machine Learning Model Applied

The following machine learning models were trained:
* Avarage Model;
* Linear Regression Model;
* Linear Regression Regularized Model - Lasso
* Random Forest Regressor;
* XGBoost Regressor.

All of them were cross-validated

# 5. Machine Learning Model Performance

The models "Random Forest Regressor" and "XGBoost Regressor" presented a better generalizability performance than the other models, but due to storage issues, the "XGBoost Regressor" was chosen. The most adequate graphs that exhibit the performance of the model in this table and graphic bellow, showing us the MAPE error of the model.

<img src="/images/model_performance.png" height="450" width="723"> ## editar

The trained (cross-validated and fine tuned) model was also applied on a dataset of potential customers who did not participate in the initial poll.


# 6. Business Results

# 7. Conclusions

# 8. Lessons Learned

* The exploratory data analysis provides important insights to the business problem, many of which contradict the initial hypothesis. This information is valuable for the understanding of business and for planning future actions. This step also provides a preview of the result of the feature selection step.
* This predict problem was solved by using a Regression adaptaded to Time-Series method
* The choice of machine learning model used must consider the generalizability of the model, but also the cost of its deployment.


# 9. Next Steps to Improve

# 10. References 


# LICENSE

# All Rights Reserved - Comunidade DS 2021
