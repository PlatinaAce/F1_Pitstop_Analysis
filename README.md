# F1_Pitstop_Analysis
25-1 Data Science Term Project - Team 12

## Overview
This repository provides a comprehensive data analysis pipeline for evaluating Formula 1 pitstop strategies using machine learning techniques. It includes preprocessing methods, model training, testing, evaluation, and identification of optimal strategy combinations. The analysis is performed on a cleaned dataset covering F1 seasons from 2011 to 2024.

## Features
### Data Preprocessing
The preprocessing function integrates:
- Data Scaling Methods
  - RobustScaler (for handling outliers)
  - StandardScaler
  - MinMaxScaler
- Categorical Feature Encoding
  - Label Encoding for categorical features like Circuit and Constructor.

## Learning Model Training and Testing
Supports multiple machine learning algorithms including:
- Random Forest Classifier
- Extra Trees Regressor (for feature importance evaluation)
Each algorithm's hyperparameters are explored comprehensively to identify optimal configurations.

## Evaluation Methods
The pipeline employs robust evaluation strategies:
- Stratified K-Fold Cross-Validation
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
