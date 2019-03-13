---
title: "Practical ML. Final project"
output: github_document
---



## *Practical machine learning final project*

The goal of this project is to predict the manner in which 6 participants did the exercise, i.e. "classe" variable in the training set. This report describes data preparation, model build and cross validation processes. We will also use this prediction model to predict 20 different test cases.

## Data preparation

Required libraries and datasets are uploaded.


```r
library(data.table)
library(caret)
library(rpart)
library(randomForest)

testData <- read.table("D:/Coursera/Data science/Practical ML/Week 4/Project/pml-testing.csv", header=TRUE, sep=",")
trainData <- read.table("D:/Coursera/Data science/Practical ML/Week 4/Project/pml-training.csv", header=TRUE, sep=",")
```

Training dataset contains around 20k rows and 160 variables.


```r
dim(trainData)
```

```
## [1] 19622   160
```

All columns with at least one NA/blank/#DIV/0! will be removed. Also columns with unnecessary information (first 7 columns).



```r
tmp <- as.data.table(trainData)
tmp[tmp == '' | tmp == ' ' | tmp == '#DIV/0!'] <- NA

trainData <- as.data.frame(tmp)
remove_NA = colnames(trainData)[ apply(trainData, 2, anyNA) ]

train_dt =  trainData[ ,!(names(trainData) %in% remove_NA)]
train_dt = train_dt[ , -c(1:7) ]
```

Creating a validation dataset, since testing dataset does not have 'classe' variable.


```r
Index = createDataPartition(train_dt$classe, p=0.7, list = FALSE)
train_dt <- train_dt[Index,]
validate_dt <- train_dt[-Index,]
```

Now we have reduced the number of variables until 53.

## Model building

We will train the data by testing *random forest decision trees (rf)* and *decision trees with CART (rpart)*. In below result we use *repeatedcv* method to divide our dataset into 3 folds cross-validation and repeat only 1 time. I will use validation set for back testing. Here *mtry*: Number of variable is randomly collected to be sampled at each split time.

The model will generate 3 random values of mtry at each time tunning, i.e. *tuneLength*  = 3.


```r
control <- trainControl(method='repeatedcv', 
                        number=3, 
                        repeats=1,
                        search = 'random')
set.seed(1)

mod_rf <- train(classe ~ ., train_dt, 
                method = "rf", 
                metric = 'Accuracy',
                tuneLength  = 3, 
                trControl = control)
```


```r
print(mod_rf)
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold, repeated 1 times) 
## Summary of sample sizes: 9157, 9159, 9158 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    3    0.9882072  0.9850800
##   15    0.9903182  0.9877519
##   26    0.9866783  0.9831478
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 15.
```


```r
set.seed(5)

mod_rpart <- train(classe ~ ., train_dt_fit, 
                method = "rpart", 
                metric = 'Accuracy',
                tuneLength  = 3, 
                trControl = control)
```


```r
print(mod_rpart)
```

```
## CART 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold, repeated 1 times) 
## Summary of sample sizes: 9158, 9158, 9158 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa    
##   0.00427220  0.8115309  0.7614612
##   0.01332520  0.6965859  0.6164526
##   0.01719052  0.6835554  0.6000400
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.0042722.
```

We will use validation dataset to compare actuals vs. predictions.


```r
predict_rf = predict(mod_rf, validate_dt_fit[ , -53 ])
confusionMatrix(predict_rf, validate_dt_fit$classe) 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1176    0    0    0    0
##          B    0  780    0    0    0
##          C    0    0  710    0    0
##          D    0    0    0  699    0
##          E    0    0    0    0  750
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9991, 1)
##     No Information Rate : 0.2858     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2858   0.1896   0.1725   0.1699   0.1823
## Detection Rate         0.2858   0.1896   0.1725   0.1699   0.1823
## Detection Prevalence   0.2858   0.1896   0.1725   0.1699   0.1823
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
predict_rpart = predict(mod_rpart, validate_dt_fit[ , -53 ])
confusionMatrix(predict_rpart, validate_dt_fit$classe) 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1059   84   10   32   15
##          B   49  553   59   63   58
##          C   14   73  566   24   22
##          D   47   40   46  564   60
##          E    7   30   29   16  595
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8109          
##                  95% CI : (0.7986, 0.8228)
##     No Information Rate : 0.2858          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.7607          
##  Mcnemar's Test P-Value : 5.023e-10       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9005   0.7090   0.7972   0.8069   0.7933
## Specificity            0.9520   0.9313   0.9609   0.9435   0.9756
## Pos Pred Value         0.8825   0.7072   0.8097   0.7450   0.8789
## Neg Pred Value         0.9599   0.9319   0.9578   0.9598   0.9549
## Prevalence             0.2858   0.1896   0.1725   0.1699   0.1823
## Detection Rate         0.2574   0.1344   0.1375   0.1371   0.1446
## Detection Prevalence   0.2916   0.1900   0.1699   0.1840   0.1645
## Balanced Accuracy      0.9263   0.8202   0.8791   0.8752   0.8845
```

## Predicting 20 test cases 

We will use *random forest* method since it provided better accuracy. The risk of overestimation.


```r
predict = predict(mod_rf, testData)
print(predict)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
