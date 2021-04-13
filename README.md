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
- [2. The Dataset](#2-the-dataset)
- [3. Solution Strategy](#3-solution-strategy)
- [4. Top 3 Data Insights](#4-top-3-data-insights)
- [5. Machine Learning Model Applied](#5-machine-learning-model-applied)
- [6. Machine Learning Model Performance](#6-machine-learning-model-performance)
- [7. Business Results](#7-bussines-results)
- [8. Conclusion](#8-conclusion)
- [9. Lessons Learned](#9-lessons-learned)
- [10. Next Steps to Improve](#10-next-steps-to-improve)
- [11.References](#11-references)

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

# 2. The Dataset

The dataset has 1017209 rows and 17 columns that represets features which explain behaviors of the target variable **Sales**.

* Id - an Id that represents a (Store, Date) duple within the test set
* Store - a unique Id for each store
* Sales - the turnover for any given day (this is what you are predicting)
* Customers - the number of customers on a given day
* Open - an indicator for whether the store was open: 0 = closed, 1 = open
* StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
* SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
* StoreType - differentiates between 4 different store models: a, b, c, d
* Assortment - describes an assortment level: a = basic, b = extra, c = extended
* CompetitionDistance - distance in meters to the nearest competitor store
* CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
* Promo - indicates whether a store is running a promo on that day
* Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
* Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
* PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

The dataset was split into training and validation sets.

The training dataset represets the sales between **2013-01-01 to 2015-06-19**

The validation dataset represnts the last 6 weeks of sales which corresponde to the date **2015-06-19 to 2015-07-31**

Since we validate our trained model, we can send it to "production", and the managers can access the sales predictions for the next six weeks.

# 3. Solution Strategy

The strategy adopted was the following:

**Step 01. Data Description:**  I searched for NAs, checked data types (and adapted some of them for analysis) and presented a statistical description.

**Step 02. Feature Engineering:** New features were created to make possible a more thorough analysis.

**Step 03. Data Filtering:**  Entries containing no information or containing information which does not match the scope of the project were filtered out.

**Step 04. Exploratory Data Analysis:**  I performed univariate, bivariate and multivariate data analysis, obtaining statistical properties of each of them, correlations and testing hypothesis (the most important of them are detailed in the following section).

**Step 05. Data Preparation:** This step is necessary both for feature selection and for the machine learning models. Regarding the data types, numerical data was rescaled and categorical data was encoded.

**Step 06. Feature Selection:** The statistically most relevant features were selected using the Boruta package. 
In the next steps, the machine learning models trained by using the features selected by Boruta presented a better generalizability performance.

**Step 07. Machine Learning Modelling:** Some machine learning models were trained. The one that presented best results after cross-validation went through a further stage of hyperparameter fine tunning to optimize the model's generalizability.

**Step 08. Hyperparameter Fine Tunning:** Found the best parameters that maximize the learning iof the model. The best parameters were found by testing a set of parameters iteratively - the set that best suit was the chosen one.

**Step 09. Convert Model Performance to Business Values:** In this step the model were analyzed from a business perpective, translating the errors into business values.

**Step 10. Deploy Model to Production:** The model were deployed on a cloud environment to make possible that other stakeholders and services access its results.

# 4. Exploratory Data Analysis and Best Insights
## 4.1 Univariate Analysis
* Numerical variables:
The histograms below show us the distribution of all the numerical features (old and new features created after feature engineering).
![](/img/feat_histogram.png)

The mind map below shows us the main factors that can contribute to to predict the target variable Sales.
Both the mind map and the available data from the dataset will be the basis to create a hypothesis list.
The hypothesis list better suit us as a guide for the Exploratory Data Analysis (EDA), which aims to better understand the general data and features properties, and generate business insights as well.

![](/img/DAILY_STORE_SALES_HYPOTESES.png)

In short:
  **All variables don't follow a normal distribution**
- **day**: There are specific days which has almost double sales data points than others (day)
- **month**: More sales data points on the first semester
- **year**: Less data points in 2015 (compared to other years)
- **customers**: Resemble a poisson distribution.
- **week_of_year**: A boom of sales data points during the first weeks of the year
- **day_of_week**: less sales data points on sundays
- **is_weekday**: more sales data points on weekdays
- **school_holiday**: more sales data points on regular days
- **competition_distance**: more sales data points for stores with closer competitors
- **competition_open_since_month**: more sales data points for competitors which entered competition on April, July, September
- **competition_since_month**: no relevant info retrieved
- **is_promo2**: more sales data points for 0
- **competition_open_since_year, promo2, promo, promo_since, promo_time**: no relevant info retrieved


* Categorical variables:

![](/img/categorical_variables.png)

In short:
- state_holiday: more sales data points on public_holidays than other holidays. Easter and Christmas are similar
- store_type: More sales data points for store_type a. Less stores b
- assortment: Less sales data points for assortment of type 'extra'

## 4.2 Bivariate Analysis

Bellow are the assumptions that I created as hypothesis to a better data undestanding and and to generate bussiness insights:

**Hypothesis 02: Stores with near competitors sell less**

**FALSE** Stores with near competitors sell more.

  * In the scatterplot graphic we can see the most part of the data concentrated within the range distance of 0 - 20000 meters;
  * Barplot show us that the revenue is higher for stores with close competitors
  * The heatmap show us a negative correlation, telling us the feature is relevant but not so much.

![](/img/H2.png)

**Hypothesis 03: Stores with longer competitors should sell more**

**FALSE** Stores with longer competitors sell less!.

  * The variable competition_since_month tells us since when a Rossmann store has started facing competitors (in months). Note that negative values mean that competition hasn't started yet.
  * Stores with new competitors sell more.

![](/img/H3.png)

**Hypothesis 12: Stores should sell less during school holidays**

**TRUE** stores sell less during school holidays. Except in August they sell more

  * school_holiday takes values of 0 (if regular day) and 1 (if school holiday);
  * Stores sell less during school holidays except in August, assuming that school break in Europe influences it.

![](/img/H12.png)

**Hypothesis Summary**:

![](/img/hypothesis_resume.png)

## 4.3 Multivariate Analysis

![](img/multi_analysis_numeric.png)

**1. Target variable & independent variables (predictors)**
* **Variables with positive correlation with sales**:

  * **Strong**: `customers`
  * **Medium**: `promo`
  * **Weak**: `is_weekday`, `promo2_since_year`
  
* **Variables with negative correlation with sales**:

  * **Strong**: -
  * **Medium**: -
  * **Weak**: `promo2`, `day_of_the_week` 

# 5. Machine Learning Model Applied

The following machine learning models were trained:
* Mean of the target variable (baseline) Model;
* Linear Regression Model;
* Linear Regression Regularized Model - Lasso
* Random Forest Regressor;
* XGBoost Regressor.

All of them were cross-validated

# 6. Machine Learning Model Performance

The **Random Forest Regressor** and the **XGBoost Regressor** were the best model performers at both cycles, with a Mean Average Percentage Error (MAPE) of 7% and 9%, respectively. Since the XGBoost Regressor is known to train data fastly than random forest algorithms (and the model performance is not too different), **we used the XGBoost regressor as the main machine learning model for the project**.

![](img/models_results_cv.png)

The trained (cross-validated and fine tuned) model was also applied on a dataset of potential customers who did not participate in the initial poll.

Using the optimal set of parameters, we obtained the following results with the XGBoost model:
![](img/xgb_tuned.png)

which had **a MAPE improvement of ~4.2%.**


# 7. Business Results

* Considering all Rossmann stores, we would have **a total predicted sales for the next six weeks of \$284,153,920**, being \$283,772,779 for the worst scenario sales prediction, and \$284,535,044 for the best scenario. Scenarios were created to reflect MAPE variations. 

  - The XGBoost model performed quite well in Rossmann stores except for three stores with MAPE above 14%:

![](img/worst_best_scenarios.PNG)

![](img/total_performance_table.PNG)

* Usually, the business has the final word on how permissible these error percentages can be. However, the model performs fairly well for most of the stores with a MAPE of ~5%. Since we have created a business case for this project, we will fictionally consider that the business has approved the model predictions.

*The line plot below shows that predictions (in orange) were fairly on par with the observed sales values (in blue) across the last six weeks of sales represented by the validation data. 

*The following graph shows the error rate (the ratio between prediction values and observed values) across six weeks of sales. **The model performs fairly well since it doesn't achieve higher error rates.** The 3rd and 5th weeks were the ones that the model performed not so well compared to other weeks:

*One of the premises for a good machine learning model is to have a normal-shaped distribution of residuals with mean zero. In the following graph, we can observe that the **errors are centered around zero, and its distribution resembles a normal, bell-shaped curve.**

*The following graph is a scatterplot with predictions plotted against the error for each sales day. Ideally, we would have all data points concentrated within a "tube" since it represents low error variance across all values that sales prediction can assume:

![](img/ML_model_performance.png)

*The machine learning model that predicts sales for Rossmann stores was deployed, and put it into production using Heroku's plataform, a PaaS that enables developers to build, run, and operate applications entirely in the cloud.

*At the end, Rossmann stakeholders will be able to access predictions with a Telegram Bot on their smartphones.

  Below, the production arthitecture used is this project:3
  
  ![](img/production_chart.png)

  The architecture works like this: (1) a user texts the store number it wishes to receive sales prediction to a Telegram Bot; (2) the Rossmann API (rossmann-bot.py) receives the request and retrieve all the data related to that store number from the test dataset; (3) the Rossmann API sends the data to Handler API (handler.py); (4) the Handler API calls the data preparation (Rossmann.py) to shape the raw data and generate predictions using the trained XGBoost model; (5) the API returns the prediction to Rossmann API; (6) the API returns the total sales prediction for a specific store + a graph of sales prediction across the next six weeks to the user on Telegram:
  
 To access the application, you can add the Telegram Bot @RossmannBot and request predictions.
  

# 8. Conclusions

In this project, all necessary steps to deploy a complete Data Science project to production were taken. Using one CRISP-DM project management methodology cycle, a satisfactory model performance was obtained by using the XGBoost algorithm to predict sales revenue for Rossmann stores up to 6 weeks in advance, and useful business information was generated during the exploratory data analysis section. Due to this, the project met the criteria of finding a suitable solution for the company's stakeholders to access sales predictions on a smartphone application.

# 9. Lessons Learned

* The exploratory data analysis provides important insights to the business problem, many of which contradict the initial hypothesis. This information is valuable for the understanding of business and for planning future actions. This step also provides a preview of the result of the feature selection step.
* This predict problem was solved by using a Regression adaptaded to Time-Series method
* The choice of machine learning model used must consider the generalizability of the model, but also the cost of its deployment.


# 10. Next Steps to Improve

# 11. References 
1. stores.csv from Kaggle (https://www.kaggle.com/c/rossmann-store-sales/data?select=store.csv)
2. train.csv from Kaggle (https://www.kaggle.com/c/rossmann-store-sales/data?select=train.csv)
3. test.csv from Kaggle (https://www.kaggle.com/c/rossmann-store-sales/data?select=test.csv)

# LICENSE
MIT LICENSE
# All Rights Reserved - Comunidade DS 2021
