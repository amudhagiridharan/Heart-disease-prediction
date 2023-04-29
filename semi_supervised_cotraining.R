#Co training with NNET and LDA with multi class
#tried using the normal params to find the severity of heart disease
# load required libraries
library(dplyr)
library(caret)
library(e1071)
library(mlbench)
library(xgboost)
library(MASS)
library(ggplot2)
library(forcats)
library(rsample)
library(tidyverse)
library(tidymodels)
library(gridExtra)
library(pROC)
library(tidyverse)
library(caret)
library(nnet)

# load cleveland_ssl dataset
heart12 <- read.csv(file = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header = FALSE)

names = c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "Class")
colnames(heart12) <- names


heart12$ca <- as.integer(heart12$ca)
heart12$thal <- as.integer(heart12$thal)
heart12 <- heart12 %>% drop_na()

str(heart12)
heart12$Classorg <- heart12$Class

heart12 <- heart12 %>%
  mutate(sex = as.factor(sex)) %>%
  mutate(cp = as.factor(cp)) %>%
  mutate(fbs = as.factor(fbs)) %>%
  #     mutate(thalach = as.factor(thalach)) %>%
  mutate(restecg = as.factor(restecg)) %>%
  mutate(exang = as.factor(exang)) %>%
  mutate(slope = as.factor(slope)) %>%
  mutate(thal = as.factor(thal)) %>%
  mutate(Class = as.factor(Class))%>%
           mutate(Classorg = as.factor(Classorg))



# Read in the hungary data
hungary <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data", header = FALSE)
# Replace "?" with NA
hungary[hungary == "?"] <- NA
colnames(hungary) <- names

# Read in the va data
va <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data", header = FALSE)
# Replace "?" with NA
va[va == "?"] <- NA
colnames(va) <- names

# Read in the swiss data
swiss <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data", header = FALSE)
# Replace "?" with NA
swiss[swiss == "?"] <- NA
colnames(swiss) <- names

#heart12_subset <- heart12[c( "age" , "sex"  , "trestbps" , "chol" , "fbs","Class","Classorg")]
n_known <- 13
known_idx <- c(sample(which(heart12$Class == 1), nrow(heart12[heart12$Class==1,])/5),
               sample(which(heart12$Class == 2), nrow(heart12[heart12$Class==2,])/5),
               sample(which(heart12$Class == 3), nrow(heart12[heart12$Class==3,])/5),
               sample(which(heart12$Class == 4), nrow(heart12[heart12$Class==4,])/5),
               sample(which(heart12$Class == 0), nrow(heart12[heart12$Class==0,])/5))

labeled <- heart12[known_idx, ]
heart_unlabeled<-heart12[-known_idx,]

# split the data into labeled and unlabeled sets
unlabeled <- rbind(hungary,swiss,va)
unlabeled$Classorg <- unlabeled$Class
unlabeled <- rbind(unlabeled,heart_unlabeled)
unlabeled$ca <- as.integer(unlabeled$ca)
unlabeled$thal <- as.integer(unlabeled$thal)
unlabeled$chol <- as.numeric(unlabeled$chol)
unlabeled$trestbps <- as.numeric(unlabeled$trestbps)
unlabeled <- unlabeled[c( "age" , "sex"  , "trestbps" , "chol" , "fbs","Class","Classorg")] %>% drop_na()

unlabeled$Class <- NaN

unlabeled <- unlabeled %>%
  mutate(sex = as.factor(sex)) %>%
  mutate(fbs = as.factor(fbs)) %>%
  mutate(Class = as.factor(Class))%>%
           mutate(Classorg = as.factor(Classorg))

labeled <- labeled[c( "age" , "sex"  , "trestbps" , "chol" , "fbs","Class","Classorg")]


unlabeled$Class <- factor(unlabeled$Class, levels = levels(heart12$Class))
labeled$Class <- factor(labeled$Class, levels = levels(heart12$Class))

cat("Row count of labeled:", nrow(labeled))
cat("Row count of unlabeled:", nrow(unlabeled))

# create two classifiers (NNET, LDA)
M1 <- nnet::multinom(Class ~ age +sex  + trestbps + chol + fbs  , data = labeled, trace = FALSE)
M2 <- lda(Class ~ age +sex  + trestbps + chol + fbs  , data = labeled)

# co-train the classifiers on the unlabeled data
n_iter <- 1300
#i=1
for (i in 1:n_iter) {
  # randomly select half of the unlabeled data
  if  (nrow(unlabeled) > 2){
    
      if (nrow(unlabeled) > 1) {
        unlabel1_idx <- sample(nrow(unlabeled), size = nrow(unlabeled) / 2)
      } else {
        unlabel1_idx <- sample(nrow(unlabeled), size =1)
      }
      unlabel2_idx <- setdiff(seq_len(nrow(unlabeled)), unlabel1_idx)
      #cat("Row count of labeled:", nrow(labeled))
      #cat("Row count of unlabeled:", nrow(unlabeled))
      # use each classifier to predict labels for the half of the unlabeled data
      preds1 <- M1 %>% predict(unlabeled[unlabel1_idx, ],type = "prob")
      preds2 <- predict(M2, newdata = unlabeled[unlabel2_idx, ])
      # cat("\ndim of preds1:", dim(preds1))
      # cat("\ndim of preds2:", dim(preds2))
      new_idx1<-""
      new_idx2<-""
      # add the most confident predictions to the labeled data
      if (length(dim(preds1)) == 2 ) {
        new_idx1 <- unlabel1_idx[apply(preds1, 1, max)>(0.999999-0.0005*max(0,(i-50)))]
      }
      #cat(i)
      if (length(new_idx1) > 0 & (new_idx1[1] != "") ) {
        print("inside loop 1")
        # if ((new_idx1 != "")   | (i==1)) 
        #   {
          unlabeled[new_idx1, ]$Class <- M1 %>% predict(unlabeled[new_idx1, ],type = "class")
          # }
      }
     # new_idx2 <- unlabel2_idx[which.max(format(preds2[unlabel2_idx]$posterior, scientific = FALSE, digits = 3))>0.99]
    
       new_idx2 <- unlabel2_idx[apply(preds2$posterior,1,max)> (0.999999-0.0005*max(0,(i-50)))]
     
       if (! is_empty(new_idx2) ) {
        unlabeled[new_idx2, ]$Class <- preds2$class[new_idx2]
      }
      

     # unlabeled <- unlabeled[-c(new_idx1, new_idx2), ]
      if (! is_empty(new_idx1) ) {
        if (new_idx1[1] == ""){
          new_idx1 = NULL
        }
      }
      if (! is_empty(new_idx2) ) {
        if (new_idx2[1] == ""){
          new_idx2 = NULL
        }
      }
      labeled <- rbind(labeled, unlabeled[c(new_idx1, new_idx2), ])
      if (all(!is.na(new_idx1)) & all(!is.na(new_idx2)) & nrow(unlabeled[-c(new_idx1, new_idx2), ]) >0) {
        unlabeled <- unlabeled[-c(new_idx1, new_idx2), ]
      }
      
      cat(" \n i=", i, " labled =", nrow(labeled), " unlabled = ",nrow(unlabeled))
      
      # retrain the classifiers on the expanded labeled data
      M1 <- nnet::multinom(Class ~ age +sex  + trestbps + chol + fbs  , data = labeled, trace = FALSE)
      M2 <- lda(Class ~ age +sex  + trestbps + chol + fbs  , data = labeled)
  }
  else{
 
    break
  }
  
}

if (nrow(unlabeled) > 0){
  unlabeled$Class <- M1 %>% predict(unlabeled,type = "class")
  labeled <- rbind(labeled, unlabeled[unlabeled$Class %in% c("0","1","2","3","4"),])
  unlabeled <- unlabeled[!unlabeled$Class %in% c("0","1","2","3","4"),]
}



confusionMatrix(factor(labeled$Class),
                factor(labeled$Classorg),
                mode = "everything")
