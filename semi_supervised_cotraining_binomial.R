#Co training with NNET and LDA for binomial classificaiton
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
library(nnet)
# load cleveland_ssl dataset
heart12 <- read.csv(file = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header = FALSE)
names = c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "Class")
colnames(heart12) <- names

#let us make the class binomial
heart12 <- heart12 %>%
  mutate(Class = case_when(Class == 0 ~ 0,
                                   (Class > 0 ~ 1)))

heart12$ca <- as.integer(heart12$ca)
heart12$thal <- as.integer(heart12$thal)
heart12 <- heart12 %>% drop_na()
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
  mutate(Class = as.factor(Classorg))


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


# consolidate the test data for unlabeled set
unlabeled <- rbind(hungary,swiss,va)
unlabeled$ca <- as.integer(unlabeled$ca)
unlabeled$thal <- as.integer(unlabeled$thal)
unlabeled$chol <- as.numeric(unlabeled$chol)
unlabeled$trestbps <- as.numeric(unlabeled$trestbps)

#let us make the class binomial
unlabeled <- unlabeled %>%
  mutate(Class = case_when(Class == 0 ~ 0,
                           (Class > 0 ~ 1)))

unlabeled <- unlabeled[c( "age" , "sex"  , "trestbps" , "chol" , "fbs","Class")] %>% drop_na()
unlabeled$Classorg <- unlabeled$Class
unlabeled$Class <- NaN

unlabeled <- unlabeled %>%
  mutate(sex = as.factor(sex)) %>%
  mutate(fbs = as.factor(fbs)) %>%
  mutate(Class = as.factor(Class))%>%
  mutate(Classorg = as.factor(Classorg))

#filter the labeled data from train data
labeled <- heart12[c( "age" , "sex"  , "trestbps" , "chol" , "fbs","Class","Classorg")]

#sample the train data for labled observations
n_known <- 5
known_idx <- c(sample(which(heart12$Class == 1), n_known),
               sample(which(heart12$Class == 0), n_known))


unlabeled$Class <- factor(unlabeled$Class, levels = levels(heart12$Class))
labeled$Class <- factor(labeled$Class, levels = levels(heart12$Class))
#labeled <- labeled[known_idx, ]
cat("Row count of labeled:", nrow(labeled))
cat("Row count of unlabeled:", nrow(unlabeled))




# create two classifiers (SVM, LDA)
M1 <-  e1071::svm(Class ~ age +sex  + trestbps + chol, data= labeled, kernel="radial",probability = T)
M2 <- lda(Class ~ age +sex  + trestbps + chol + fbs  , data = labeled,class=binomial )

# co-train the classifiers on the unlabeled data
n_iter <- 1200
for (i in 1:n_iter) {
  # randomly select half of the unlabeled data
  if  (nrow(unlabeled) > 2){
    
      if (nrow(unlabeled) > 1) {
        unlabel1_idx <- sample(nrow(unlabeled), size = nrow(unlabeled) / 2)
      } else {
        unlabel1_idx <- sample(nrow(unlabeled), size =1)
      }
      unlabel2_idx <- setdiff(seq_len(nrow(unlabeled)), unlabel1_idx)
      cat("Row count of labeled:", nrow(labeled))
      cat("Row count of unlabeled:", nrow(unlabeled))
      # use each classifier to predict labels for the half of the unlabeled data
      preds1 <- predict(M1,unlabeled[unlabel1_idx, ])
      preds2 <- predict(M2, newdata = unlabeled[unlabel2_idx, ])
      # cat("\ndim of preds1:", dim(preds1))
      # cat("\ndim of preds2:", dim(preds2))
      new_idx1<-""
      new_idx2<-""
      # add the most confident predictions to the labeled data
      if (length(dim(preds1)) == 2 ) {
        new_idx1 <- unlabel1_idx[apply(preds1$posterior,1,max)> (0.999999-0.0005*max(0,(i-50)))]
      }
      cat(i)
      if (length(new_idx1) > 0 & (new_idx1[1] != "") ) {
        print("inside loop 1")
        # if ((new_idx1 != "")   | (i==1)) 
        #   {
          unlabeled[new_idx1, ]$Class <- predict(M1,unlabeled[new_idx1, ])
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
      
      cat("\n i=", i, "\n labled =", nrow(labeled), "\n unlabled = ",nrow(unlabeled))
      
      # retrain the classifiers on the expanded labeled data
      M1 <-  e1071::svm(Class ~ age +sex  + trestbps + chol, data= labeled, kernel="radial",probability = T)
      M2 <- lda(Class ~ age +sex  + trestbps + chol + fbs  , data = labeled,class=binomial )
      
  }
  else{
 
    break
  }
  
}

if (nrow(unlabeled) > 0){
  unlabeled[new_idx2, ]$Class <- preds2$class[new_idx2]
  labeled <- rbind(labeled, unlabeled[unlabeled$Class %in% c("0","1"),])
  unlabeled <- unlabeled[!unlabeled$Class %in% c("0","1"),]
}


# labeled_ordered <- rbind(labeled,unlabeled) %>%
#   arrange(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
# 
# heart_ordered <- heart12 %>%
#   arrange(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

# table(heart_ordered$Class , labeled_ordered$Class )
# 
# (labeled_ordered$Class == heart_ordered$Class)


confusionMatrix(factor(labeled$Class),
                factor(labeled$Classorg),
                mode = "everything")
