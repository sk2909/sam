import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator

# Load the heart disease dataset
heart_disease_data = pd.read_csv('heartdisease.csv')
heart_disease_data = heart_disease_data.replace('?', np.nan)

# Display sample data and attributes
print('Sample instances from the dataset:')
print(heart_disease_data.head())
print('\nAttributes and datatypes:')
print(heart_disease_data.dtypes)

# Define the Bayesian network structure
model = BayesianModel([
    ('age', 'heartdisease'),
    ('gender', 'heartdisease'),
    ('exang', 'heartdisease'),
    ('cp', 'heartdisease'),
    ('heartdisease', 'restecg'),
    ('heartdisease', 'chol')
])

# Fit the model to the data
print("\nLearning CPD using Maximum Likelihood Estimator")
model.fit(heart_disease_data, estimator=MaximumLikelihoodEstimator)

# Perform inference
print("\nInferencing with Bayesian Network:")
inference = VariableElimination(model

                                )

# Query the model
# 1. Probability of heart disease given evidence of restecg
print('\nProbability of Heart Disease given age evidence:')
q1 = inference.query(variables=['heartdisease'], evidence={'age': 60})
print(q1)

# 2. Probability of heart disease given evidence of cp
print('\nProbability of Heart Disease given cp evidence:')
q2 = inference.query(variables=['heartdisease'], evidence={'cp': 2})
print(q2)