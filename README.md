# Credit Risk Analysis

## Overview of the Analysis

Using machine learning models, this assessment performs an analysis of the credit worthiness of a peer-to-peer lending services company via the examination of a sample of its loan book. 

The purpose of the analysis is to determine how each model can identify high-risk loan customers via two modeling techniques, both using the `Logistic Regression` classifier with different training subsets of data.

The data consists of 77,536 loans including the:
  * Loan size
  * Interest rate
  * Borrower income
  * Debt to income ratio
  * Number of accounts
  * Derogatory marks (as a number)
  * Total debt and
  * Loan status

The model will predict the loan status based on the above features.  After designating `loan status` as the target label, the data will be split between a `training` set and a `testing` set with the `training` set roughly being `25%` of the original.

The stages of the machine learning process include an initial run using the `Logistic Regression` classifier algorithm to train the model.  The results of this initial run are then reviewed by examining the `confusion matrix` and its `classification report`.  This will then be followed by another run after the training data is `oversampled` in order to get a better representation of the `high-risk` group.  The performance of both models will then be compared.

## Results

### Machine Learning Model 1:

`Model 1` uses the training data obtained from the `train_test_split` function of the `scikit-learn` library with the following characteristics:
||Healthy loans|High-risk loans|
|-|-------------|---------------|
|Count|75036|2500|

The balanced accuracy score for model 1 was: `0.9520479254722232`.

#### Confusion Matrix `model 1`

||Positive|Negative|
|-|------|-----|
|Positive|18663|102|
|Negative|56|563


#### Classification report `model 1`

||Precision|Recall|Specificity|F1|Geometric Mean|Indexed Balanced Accuracy|Sup|
|-|-|-|-|-|-|-|-|
|Healthy Loans|1.00|0.99|.091|1.00|0.95|0.91|18765
|High-Risk Loans|0.85|0.91|0.99|0.88|0.95|0.91|619|
|Avg/Total|0.99|0.99|0.91|0.99|0.95|0.91|19384


### Machine Learning Model 2:
`Model 2` oversamples the training data obtained to augment the `high-risk` group in order to match the `healthy loans` group count in an effort to improve the performance.

||Healthy loans|High-risk loans|
|-|-------------|---------------|
|Count|56271|56271|

The balanced accuracy score for `model 2` was `0.9936781215845847`.

#### Confusion Matrix `model 2`
||Positive|Negative|
|-|------|-----|
|Positive|18649|116|
|Negative|4|615

## Summary

In summary, although model 2's `precision` ratio for `high-risk loans` dropped by `0.01`, i.e. `model 2` lost its ability not to label as high-risk a loan is actually low-risk by a very small amount, its recall ratio increased by `0.08` indicating that it has increased its ability to find more `high-risk` loans over `model 1`.  I would therefore recommend `model 2`'s performance over `model 1`.
