# Install libraries if not exist

if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(data.table)) install.packages("data.table")
if(!require(tidyr)) install.packages("tidyr")
if(!require(stringr)) install.packages("stringr")
if(!require(forcats)) install.packages("forcats")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(funModeling)) install.packages("funModeling")

# Loading all needed libraries

library(dplyr)
library(tidyverse)
library(dslabs)
library(caret)
library(kableExtra)
library(tidyr)
library(stringr)
library(forcats)
library(ggplot2)
library(gam)
library(splines)
library(foreach)
library(mgcv)
library(nlme)
library(data.table)
library(funModeling)


# Heart Disease Data Set:
# http://archive.ics.uci.edu/ml/datasets/Heart+Disease
# https://www.kaggle.com/ronitf/heart-disease-uci/download
# Relevant Information:
# This database contains 76 attributes, but all published experiments
# refer to using a subset of 14 of them. In particular, the Cleveland
# database is the only one that has been used by ML researchers to 
# this date. The "goal" field refers to the presence of heart disease
# in the patient.
# there is a package that contains heart_desease dataset: funModeling
# but we download our own data from Kaggle

# automated downloading file "heart.csv" from my github repository

urlfile <-
  "https://raw.githubusercontent.com/aleboffa/HarvardX-Heart-Disease-Capstone/master/heart.csv"

Hearts <- read.csv(urlfile, header = TRUE) # this is our working dataset

# change for more descriptive column names

names(Hearts) <- c("age", "sex", "chest_pain_type", "resting_blood_pressure",
                   "cholesterol", "fasting_blood_sugar", "rest_ecg",
                   "max_heart_rate_achieved", "exercise_induced_angina",
                   "st_depression", "st_slope", "num_major_vessels",
                   "thallium_stress_test", "disease")

# converts disease as factor "0" or "1" to use with confusionMatrix()

Hearts <- mutate_at(Hearts, vars(disease), as.factor)

# now calculate PCA removing "disease" with correlation "1"

set.seed(1)
pca <- prcomp(Hearts %>% select(-disease), scale = TRUE, center = TRUE)

# Compute the proportion of variance explained(PVE)

pve_Hearts <- pca$sdev^2 / sum(pca$sdev^2)
cum_pve <- cumsum(pve_Hearts)     # Cummulative percent explained
pve_table <- tibble(comp = seq(1:ncol(Hearts %>% select(-disease))),
                    pve_Hearts, cum_pve)

# plot components vs PVE to see if there are correlations
# to reduce variable numbers. All have to be independent variables

ggplot(pve_table, aes(x = comp, y = cum_pve)) + 
  geom_point() + 
  geom_abline(intercept = 0.95, color = "red",
              slope = 0)   # line at 95% of cum_pve

########################################################
#  Spliting Hearts dataset in edx and validation sets
#  We will use it in the final algorithm
########################################################

set.seed(1) # if you are using R 3.5 or Microsoft R Open 3.5.3
# set.seed(1, sample.kind="Rounding") if using R 3.5.3 or later

# Validation set will be 20% of Hearts data because it is a little dataset

test_index <- createDataPartition(y = Hearts$disease,
                                  times = 1, p = 0.2, list = FALSE) 
edx <- Hearts[-test_index,]
validation <- Hearts[test_index,] # we will use it only to do final test


########################################################
#  Spliting edx dataset in train_set and test_set
#  We will use it to train ours models
########################################################

set.seed(1) # if you are using R 3.5 or Microsoft R open 3.5.3
# set.seed(1, sample.kind="Rounding") # if using R 3.5.3 or later

test_index <- createDataPartition(y = edx$disease,
                                  times = 1, p = 0.2, 
                                  list = FALSE)  # test_set 20%
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

###########################
#  Model building approach
###########################

# model list, it can be any model you want to test,
# just be sure to load the library that contains it

models <- c("glm", "lda", "naive_bayes", "svmLinear",
            "gamLoess", "qda", "knn", "kknn",
            "gam", "rf", "ranger", "wsrf", "mlp")

# trainControl function for control tuning parameters models

control <- trainControl(method = "cv",   # cross validation
                        number = 10,     # 10 k-folds or number 
                        # of resampling iterations
                        repeats = 5)

# initializing variables

data_train <- train_set       # first value for data parameter
data_test <-  test_set        # first we´ll use train and test dataset 
true_value <- test_set$disease # true outcome from test_set 

# loop to use train and test set first and edx and validation later

for(i in 1:2) {       
  fits <- lapply(models, function(model){ 
    #    print(model)  # it´s used to debug code
    set.seed(1)
    train(disease ~ .,
          method = model,
          preProcess=c("center", "scale"),   # to normalize the data
          data = data_train,
          trControl = control)
  })
  
  names(fits) <- models
  
  # to be sure that the actual value of the output
  # has not influence on the prediction
  
  vali2 <- data_test %>% select(-disease) 
  
  pred <- sapply(fits, function(object) # predicting outcome
    predict(object, newdata = vali2))
  
  # avg predicted values if equals to true values
  
  if (i == 1) acc <- colMeans(pred == true_value)
  
  data_train <- edx               # last value for data parameter
  data_test <-  validation        # last we´ll use edx and validation
  true_value <- validation$disease  # true outcome from validation set
  
}

# we choose the model with the smallest value(variation between datasets)

acc    # all accuracy values with first dataset. Train and Test set

acc2 <- colMeans(pred == true_value) # avg predicted values

acc2   # all accuracy values with last dataset. Edx and Validation set

results <- acc2 - acc # accuracy diff by model

# show results of difference on the same model for different dataset 

results

# Compute balance accuracy, sensitivity, specificity,
# prevalence with confusionMatrix

cm_validation_hearts<- confusionMatrix(as.factor(pred[,11]),
                                       validation$disease, positive = "1")
cm_validation_hearts

#############################################
# Inside Ranger Method: a Random Forest Model
#############################################

# Final Ranger model algorithm computed with principal
# edx and validation dataset

# to avoid error in confusionMatrix, we convert num 0,1 to No,Yes

levels(edx$disease) <- c("No", "Yes")        
levels(validation$disease) <- c("No", "Yes")

# to be sure that the actual value of the output
# has not influence on the prediction

vali2 <- validation %>% select(-disease)

# trainControl function for control iteration model
# we test differents parameters and choose that ones that improve accuracy

control <- trainControl(method = "cv",   # cross validation
                        number = 30)   # optimum k-folds or number 30
# of resampling iterations

# training Ranger Model

set.seed(1)
ranger_model <- train(disease ~., data = edx,
                      method = "ranger",  # ranger model
                      preProcess=c("center", "scale"),   # to normalize the data
                      trControl = control)

# predicting outcome

prediction_ranger <- predict(ranger_model, newdata = vali2)

# Check results with confusionMatrix() function and validation set

cm_ranger_model <- confusionMatrix(prediction_ranger,
                                   validation$disease, positive = "Yes")
cm_ranger_model


###############
# Results
###############

# results from several models computed with edx and validation datasets

acc2 %>% knitr::kable(col.names = c("Model    Accuracy"))

results <- (acc2 - acc) * 100 # accuracy difference by model in %

# show results of difference on the same model for different dataset 

results %>% knitr::kable(col.names = c("Accuracy difference by model in %"))

# barchart with difference between 2 datasets on the same model,
# smaller is better

barchart(results)

############
# accuracy result from our best Final Model computed with
# original edx and validation datasets

# compute final accuracy and convert proportion to percentage

acc3 <- (mean(prediction_ranger == validation$disease)) * 100

# show final accuracy

paste0("Ranger Model Accuracy: ", round(acc3, digits=2)," %")



############################################################
############################################################
# Remmember, our goal is obtain a model algorithm to predict
# Heart Disease on a given dataset, so we found the Ranger
# Model algorithm that complies with our searching:
#
# Final Ranger Model Algorithm results:
#
# Balanced Accuracy = 0.90 
# Sensitivity = 0.90 
# Specificity = 0.85
# Detection Rate = 0.49
#
############################################################