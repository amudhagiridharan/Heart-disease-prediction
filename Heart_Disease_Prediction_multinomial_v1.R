# multinomial prediction of heart disease
library(dplyr)
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


cleveland <- read.csv(file = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header = FALSE)
head(cleveland)

names = c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "heart_disease")
colnames(cleveland) <- names

# cleveland <- cleveland %>%
#   mutate(heart_disease = case_when(heart_disease == 0 ~ 0,
#                                    (heart_disease > 0 ~ 1)))

# cleveland <- cleveland %>%
#   mutate(sex = case_when(sex == 0 ~ "female",
#                          sex == 1 ~ "male")) %>%
#   mutate(cp = case_when(cp == 1 ~ "typical angina",
#                         cp == 2 ~ "atypical angina",
#                         cp == 3 ~ "non-anginal pain",
#                         cp == 4 ~ "asymptomatic")) %>%
#   mutate(fbs = case_when(fbs == 1 ~ "high",
#                          fbs == 0 ~ "low")) %>%
#   mutate(exang = case_when(exang == 0 ~ "no",
#                            exang == 1 ~ "yes"))

cleveland$ca <- as.integer(cleveland$ca)
cleveland$thal <- as.integer(cleveland$thal)
cleveland <- cleveland %>% drop_na()

str(cleveland)


cleveland <- cleveland %>%
   mutate(sex = as.factor(sex)) %>%
     mutate(cp = as.factor(cp)) %>%
    mutate(fbs = as.factor(fbs)) %>%
#     mutate(thalach = as.factor(thalach)) %>%
     mutate(restecg = as.factor(restecg)) %>%
     mutate(exang = as.factor(exang)) %>%
     mutate(slope = as.factor(slope)) %>%
     mutate(thal = as.factor(thal)) %>%
     mutate(heart_disease = as.factor(heart_disease))

# cleveland <- cleveland %>%
#   select(age, sex, cp, trestbps, chol, fbs, thalach, exang, heart_disease) %>%
#   rename("max_hr" = "thalach",
#          "exercise_angina" = "exang") %>%
#   drop_na()

glimpse(cleveland)

set.seed(03) 

trainset <- sample(1:nrow(cleveland), 0.80*nrow(cleveland))
validset <- setdiff(1:nrow(cleveland), trainset) # The remaining is used for validation
train <- cleveland[trainset,]
test <- cleveland[validset,]


# scenario 1
# neural network multinomial classification with all the attributes
print("Scenario 1 - neural network multinomial classification with all attributes")
#max accuracy
model_all <- nnet::multinom(heart_disease ~., data = cleveland[trainset,])
# Summarize the model
#summary(model_all)

# Make predictions for test
predicted.classes <- model_all %>% predict(test)
head(predicted.classes)
# Model accuracy
mean(predicted.classes == test$heart_disease)
#confusion matrix
table(predicted.classes ,test$heart_disease)


# Make predictions for train
predicted.classes <- model_all %>% predict(train)
head(predicted.classes)
# Model accuracy
mean(predicted.classes == train$heart_disease)
#confusion matrix
table(predicted.classes ,train$heart_disease)

# scenario 2
# neural network multinomial classification removing categorical vars
print("Scenario 2 - neural network multinomial classification removing categorical attributes")

model1 <- nnet::multinom(heart_disease ~ age +thalach+ trestbps + chol+ oldpeak + ca, data = cleveland[trainset,])
# Summarize the model
summary(model1)
# Make test predictions
predicted1.classes <- model1 %>% predict(test)
head(predicted1.classes)
# Model accuracy
mean(predicted1.classes == test$heart_disease)
#confusion matrix
table(predicted1.classes ,test$heart_disease)

# Make train predictions
predicted1.classes <- model1 %>% predict(train)
head(predicted1.classes)
# Model accuracy
mean(predicted1.classes == train$heart_disease)
#confusion matrix
table(predicted1.classes ,train$heart_disease)


# scenario 3
# neural network multinomial classification with significant attributes based on Chi-square test
print("Scenario 3 - neural network multinomial classification with significant attributes based on Chi-square test")

#based on chi-test
model_all <- nnet::multinom(heart_disease ~.-age-trestbps-chol-fbs-thalach, data = cleveland[trainset,])
# Summarize the model
#summary(model_all)
# Make test predictions
predicted.classes <- model_all %>% predict(test)
head(predicted.classes)
# Model accuracy
mean(predicted.classes == test$heart_disease)
#confusion matrix
table(predicted.classes ,test$heart_disease)

# Make train predictions
predicted.classes <- model_all %>% predict(train)
head(predicted.classes)
# Model accuracy
mean(predicted.classes == train$heart_disease)
#confusion matrix
table(predicted.classes ,train$heart_disease)

model_all <- nnet::multinom(heart_disease ~sex+cp+restecg+exang+oldpeak+slope+ca+thal, data = cleveland[trainset,])
# Summarize the model
summary(model_all)
# Make predictions
predicted.classes <- model_all %>% predict(test)
head(predicted.classes)
# Model accuracy
mean(predicted.classes == test$heart_disease)
#confusion matrix
table(predicted.classes ,test$heart_disease)


model_all <- nnet::multinom(heart_disease ~sex+cp+restecg+exang+oldpeak+slope+ca+thal, data = cleveland[trainset,])
# Summarize the model
#summary(model_all)
# Make predictions
predicted.classes <- model_all %>% predict(test)
head(predicted.classes)
# Model accuracy
mean(predicted.classes == test$heart_disease)
#confusion matrix
table(predicted.classes ,test$heart_disease)

# predicted.classes  
#     0   1   2   3   4
# 0 105  15   5   4   0
# 1  10   9   8   6   3
# 2   5   7   5   4   4
# 3   1   5   5   6   0
# 4   4   3   2   5   1


model_all <- nnet::multinom(heart_disease ~oldpeak+ca+thal, data = cleveland[trainset,])
# Summarize the model
#summary(model_all)
# Make predictions
predicted.classes <- model_all %>% predict(test)
head(predicted.classes)
# Model accuracy
mean(predicted.classes == test$heart_disease)
#confusion matrix
table(predicted.classes ,test$heart_disease)


#LDA

m1 <- lda (heart_disease ~ .-age-trestbps-chol-fbs-thalach , data = train)
pred <- predict (m1, test)
#confusion matrix
table(pred$class, test$heart_disease)
# Model accuracy
round(mean(pred$class == test$heart_disease),4)

#trainig acuracy
pred <- predict (m1, train)
#confusion matrix
table(pred$class, train$heart_disease)
# Model accuracy
round(mean(pred$class == train$heart_disease),4)


m1 <- lda (heart_disease ~ sex+cp+restecg+thalach+exang+oldpeak+slope+ca+thal , data = train)
pred <- predict (m1, test)
#confusion matrix
table(pred$class, test$heart_disease)
# Model accuracy
round(mean(pred$class == test$heart_disease),4)

#max accuracy
m1 <- lda (heart_disease ~ . , data = train)
pred <- predict (m1, test)
#confusion matrix
table(pred$class, test$heart_disease)
# Model accuracy
round(mean(pred$class == test$heart_disease),4)

#trainig acuracy
pred <- predict (m1, train)
#confusion matrix
table(pred$class, train$heart_disease)
# Model accuracy
round(mean(pred$class == train$heart_disease),4)



library(class)
for (k in c(1,3,5,7,10,50,100)){
  m4 <- knn(train[ ,c("sex","cp", "restecg", "thalach", "exang", "oldpeak" , "slope", "ca", "thal")],
            test[ ,c("sex","cp", "restecg", "thalach", "exang", "oldpeak" , "slope", "ca", "thal")],
            train$heart_disease, k)
  table(m4, test$heart_disease)
  print(paste("Test error rate of KNN with k =", k, "is", round(mean(m4!=test$heart_disease),4)))
}

table(m4,test$heart_disease)



model_all <- nnet::multinom(heart_disease ~., data = cleveland[trainset,])
# Summarize the model
#summary(model_all)
# Make predictions for test
predicted.classes <- model_all %>% predict(test)
head(predicted.classes)
# Model accuracy
mean(predicted.classes == test$heart_disease)
#confusion matrix
table(predicted.classes ,test$heart_disease)

pred<-predict(model_all,test, type="class")
head(pred)
table(pred ,test$heart_disease)
mean(pred == test$heart_disease)

library(e1071)
##SVM for multi classification

M1 <- e1071::svm(heart_disease ~ ., data= train, kernel="radial",probability = T)
pred <- predict(M1,test)
confusionMatrix(pred,test$heart_disease)
head(pred)
table(pred ,test$heart_disease)
mean(pred == test$heart_disease)

pred <- predict(M1,train)
confusionMatrix(pred,train$heart_disease)
head(pred)
table(pred ,train$heart_disease)
mean(pred == train$heart_disease)


#better than all
M1 <- e1071::svm(heart_disease ~ sex+cp+restecg+thalach+exang+oldpeak+slope+ca+thal, data= train, kernel="radial",probability = T)
pred <- predict(M1,test)
confusionMatrix(pred,test$heart_disease)
head(pred)
table(pred ,test$heart_disease)
mean(pred == test$heart_disease)

pred <- predict(M1,train)
confusionMatrix(pred,train$heart_disease)
head(pred)
table(pred ,train$heart_disease)
mean(pred == train$heart_disease)


