#XGB

library("xgboost")  
library("archdata") 
library("caret")    
library("dplyr")  
library("Ckmeans.1d.dp") 

cleveland1 <- read.csv(file = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header = FALSE)
names = c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "heart_disease")
colnames(cleveland1) <- names

numberOfClasses <- length(unique(cleveland1$heart_disease))

xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)
nround    <- 50 # number of XGBoost rounds
cv.nfold  <- 5


cleveland1$ca <- as.integer(cleveland1$ca)
cleveland1$thal <- as.integer(cleveland1$thal)
cleveland1 <- cleveland1 %>% drop_na()
# Make split index
train_index <- sample(1:nrow(cleveland1), nrow(cleveland1)*0.60)


# Full data set
data_variables <- as.matrix(cleveland1[,-14])
data_label <- cleveland1[ ,14]
data_matrix <- xgb.DMatrix(data = as.matrix(cleveland1), label = data_label)
# split train data and make xgb.DMatrix
train_data   <- data_variables[train_index,]
train_label  <- data_label[train_index]
train_matrix <- xgb.DMatrix(data = train_data, label = train_label)
# split test data and make xgb.DMatrix
test_data  <- data_variables[-train_index,]
test_label <- data_label[-train_index]
test_matrix <- xgb.DMatrix(data = test_data, label = test_label)

# Fit cv.nfold * cv.nround XGB models and save OOF predictions
cv_model <- xgb.cv(params = xgb_params,
                   data = train_matrix, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = FALSE,
                   prediction = TRUE)

OOF_prediction <- data.frame(cv_model$pred) %>%
  mutate(max_prob = max.col(., ties.method = "last"),
         label = train_label + 1)
head(OOF_prediction)


# confusion matrix
confusionMatrix(factor(OOF_prediction$max_prob),
                factor(OOF_prediction$label),
                mode = "everything")

mean(factor(OOF_prediction$max_prob) == factor(OOF_prediction$label))

bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = nround)

# Predict hold-out test set
test_pred <- predict(bst_model, newdata = test_matrix)
test_prediction <- matrix(test_pred, nrow = numberOfClasses,
                          ncol=length(test_pred)/numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(label = test_label + 1,
         max_prob = max.col(., "last"))
# confusion matrix of test set
confusionMatrix(factor(test_prediction$max_prob),
                factor(test_prediction$label),
                mode = "everything")

mean(factor(test_prediction$max_prob) == factor(test_prediction$label))


train_pred <- predict(bst_model, newdata = train_matrix)
train_prediction <- matrix(train_pred, nrow = numberOfClasses,
                          ncol=length(train_pred)/numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(label = train_label + 1,
         max_prob = max.col(., "last"))
# confusion matrix of train set
confusionMatrix(factor(train_prediction$max_prob),
                factor(train_prediction$label),
                mode = "everything")

mean(factor(train_prediction$max_prob) == factor(train_prediction$label))


# get the feature real names
names <-  colnames(cleveland1[,-1])
# compute feature importance matrix
importance_matrix = xgb.importance(feature_names = names, model = bst_model)
head(importance_matrix)


# plot
xgb.ggplot.importance(importance_matrix)


##*************** Chi test attributes ******************##

data_variables <- as.matrix(cleveland1[,c(-1,-4,-5,-6,-14)])
data_label <- cleveland1[ ,14]
data_matrix <- xgb.DMatrix(data = as.matrix(cleveland1), label = data_label)
# split train data and make xgb.DMatrix
train_data   <- data_variables[train_index,]
train_label  <- data_label[train_index]
train_matrix <- xgb.DMatrix(data = train_data, label = train_label)
# split test data and make xgb.DMatrix
test_data  <- data_variables[-train_index,]
test_label <- data_label[-train_index]
test_matrix <- xgb.DMatrix(data = test_data, label = test_label)

# Fit cv.nfold * cv.nround XGB models and save OOF predictions
cv_model <- xgb.cv(params = xgb_params,
                   data = train_matrix, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = FALSE,
                   prediction = TRUE)

OOF_prediction <- data.frame(cv_model$pred) %>%
  mutate(max_prob = max.col(., ties.method = "last"),
         label = train_label + 1)
head(OOF_prediction)


# confusion matrix
confusionMatrix(factor(OOF_prediction$max_prob),
                factor(OOF_prediction$label),
                mode = "everything")


bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = nround)

# Predict hold-out test set
test_pred <- predict(bst_model, newdata = test_matrix)
test_prediction <- matrix(test_pred, nrow = numberOfClasses,
                          ncol=length(test_pred)/numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(label = test_label + 1,
         max_prob = max.col(., "last"))
# confusion matrix of test set
confusionMatrix(factor(test_prediction$max_prob),
                factor(test_prediction$label),
                mode = "everything")

mean(factor(test_prediction$max_prob) == factor(test_prediction$label))


train_pred <- predict(bst_model, newdata = train_matrix)
train_prediction <- matrix(train_pred, nrow = numberOfClasses,
                           ncol=length(train_pred)/numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(label = train_label + 1,
         max_prob = max.col(., "last"))

mean(factor(train_prediction$max_prob) == factor(train_prediction$label))


# get the feature real names
names <-  bst_model$feature_names
# compute feature importance matrix
importance_matrix = xgb.importance(feature_names = names, model = bst_model)
head(importance_matrix)


# plot
xgb.ggplot.importance(importance_matrix)


