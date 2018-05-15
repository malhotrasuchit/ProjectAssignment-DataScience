Practical Machine Learning Course Project
================
May 15, 2018

Overview
--------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset)."

Data
----

The data for this project come from this source: <http://groupware.les.inf.puc-rio.br/har>. The information has been generously provided for use use in this cousera course by the authors, Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. They have allowed the use of their paper "Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

The goal of this project is to build a machine learning algorithm to predict activity quality (classe) from activity monitors.

Choosing a prediction model
---------------------------

Steps Taken - Tidy data. Remove columns with little/no data. - Create Training and test data from traing data for cross validation checking - Trial 3 methods Random Forrest, Gradient boosted model and Linear discriminant analysis - Fine tune model through combinations of above methods, reduction of input variables or similar. The fine tuning will take into account accuracy first and speed of analysis second.

``` r
library(caret)
```

    ## Warning: package 'caret' was built under R version 3.4.4

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 3.4.1

### Load data

    - Load data.
    - Remove "#DIV/0!", replace with an NA value.

``` r
# load data
train <- read.csv("pml-training.csv", na.strings=c("#DIV/0!"), row.names = 1)
test <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!"), row.names = 1)
```

Preprocessing
-------------

### Partitioning the training set

We separate our training data into a training set and a validation set so that we can validate our model.

``` r
set.seed(123456)
trainset <- createDataPartition(train$classe, p = 0.8, list = FALSE)
Training <- train[trainset, ]
Validation <- train[-trainset, ]
```

### Feature Selection

First we clean up near zero variance features, columns with missing values and descriptive fields.

``` r
# exclude near zero variance features
nzvcol <- nearZeroVar(Training)
Training <- Training[, -nzvcol]

# exclude columns with m40% ore more missing values exclude descriptive
# columns like name etc
cntlength <- sapply(Training, function(x) {
    sum(!(is.na(x) | x == ""))
})
nullcol <- names(cntlength[cntlength < 0.6 * length(Training$classe)])
descriptcol <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", 
    "cvtd_timestamp", "new_window", "num_window")
excludecols <- c(descriptcol, nullcol)
Training <- Training[, !names(Training) %in% excludecols]
```

Train Model
-----------

We will use random forest as our model as implemented in the randomForest package by Breiman's random forest algorithm (based on Breiman and Cutler's original Fortran code) for classification and regression.

``` r
library(randomForest)
```

    ## Warning: package 'randomForest' was built under R version 3.4.4

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
library(e1071)
```

    ## Warning: package 'e1071' was built under R version 3.4.4

``` r
rfModel <- randomForest(classe ~ ., data = Training, importance = TRUE, ntrees = 10)
```

Model Validation
----------------

Let us now test our model performance on the training set itself and the cross validation set.

### Training set accuracy

``` r
ptraining <- predict(rfModel, Training)
print(confusionMatrix(ptraining, Training$classe))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 4464    0    0    0    0
    ##          B    0 3038    0    0    0
    ##          C    0    0 2738    0    0
    ##          D    0    0    0 2573    0
    ##          E    0    0    0    0 2886
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9998, 1)
    ##     No Information Rate : 0.2843     
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
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

Obviously our model performs excellent against the training set, but we need to cross validate the performance against the held out set and see if we have avoided overfitting.

### Validation set accuracy (Out-of-Sample)

``` r
pvalidation <- predict(rfModel, Validation)
confusionMatrix(pvalidation, Validation$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1116    8    0    0    0
    ##          B    0  750    5    0    0
    ##          C    0    1  679    4    0
    ##          D    0    0    0  639    3
    ##          E    0    0    0    0  718
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9946          
    ##                  95% CI : (0.9918, 0.9967)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9932          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9881   0.9927   0.9938   0.9958
    ## Specificity            0.9971   0.9984   0.9985   0.9991   1.0000
    ## Pos Pred Value         0.9929   0.9934   0.9927   0.9953   1.0000
    ## Neg Pred Value         1.0000   0.9972   0.9985   0.9988   0.9991
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2845   0.1912   0.1731   0.1629   0.1830
    ## Detection Prevalence   0.2865   0.1925   0.1744   0.1637   0.1830
    ## Balanced Accuracy      0.9986   0.9933   0.9956   0.9964   0.9979

The cross validation accuracy is 99.5% and the out-of-sample error is therefore 0.5% so our model performs rather good.

Apply to final test set
-----------------------

Finally, we apply our model to the final test data. Upon submission all predictions were correct!

``` r
ptest <- predict(rfModel, test)
ptest
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

We then save the output to files according to instructions and post it to the submission page.

``` r
answers <- as.vector(ptest)

pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}

pml_write_files(answers)
```
