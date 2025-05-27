library(tidyverse)  # For data manipulation and visualization
library(GGally)     # For advanced plotting
library(DataExplorer) # For automated EDA
library(corrplot)   # For correlation plots
library(skimr)      # For data summary
library(caret)      # For model training and evaluation
library(randomForest) # For random forest models
library(pROC)       # For ROC curve analysis
library(gridExtra)  

# Load training and testing datasets
train <- read.csv("customer_churn_dataset-training-master.csv")
test <- read.csv("customer_churn_dataset-testing-master.csv")

# 2. Data Overview of training data
glimpse(train)
dim(train)
head(train)

# 3. Data Preprocessing for both datasets
# Process training data
train$Gender <- as.factor(train$Gender)
train$Subscription.Type <- as.factor(train$Subscription.Type)
train$Contract.Length <- as.factor(train$Contract.Length)
train$Churn <- as.factor(train$Churn)
train <- select(train, -CustomerID)

# Process testing data
test$Gender <- as.factor(test$Gender)
test$Subscription.Type <- as.factor(test$Subscription.Type)
test$Contract.Length <- as.factor(test$Contract.Length)
test$Churn <- as.factor(test$Churn)
test <- select(test, -CustomerID)

# Check for missing values
sum(is.na(train))
sum(is.na(test))

# Check for duplicates
train <- train %>% distinct()
test <- test %>% distinct()

missing_train <- colSums(is.na(train))
print(missing_train)  # See which columns have missing values

train <- na.omit(train)  # More aggressive than drop_na()

sum(is.na(train))

# Handle missing values in test data too
test <- na.omit(test)  # Or use a more targeted approach if preferred

# Check lengths to ensure they match
cat("Rows in test dataset:", nrow(test), "\n")


# 4. Univariate Analysis ----
skimr::skim(train)  # Only need to analyze training data


# 5. Bivariate & Multivariate Analysis ----
corrplot(cor(train %>% select_if(is.numeric)), method = "color", tl.cex = 0.8)

# 6. Visualization ----
train %>% 
  ggplot(aes(x = Subscription.Type, fill = Churn)) +
  geom_bar(position = "fill") +
  labs(title = "Churn Rate by Subscription Type")

# KDE Plot
train %>%
  ggplot(aes(x = Total.Spend, fill = Churn)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot: Total Spend by Churn")

# 7. Logistic Regression ----
log_model <- glm(Churn ~ ., data = train, family = "binomial", 
                 control = glm.control(maxit = 50))
log_preds <- predict(log_model, test, type = "response")
log_class <- ifelse(log_preds > 0.5, 1, 0) %>% as.factor()
log_class <- factor(log_class, levels = levels(test$Churn))


# Check that the test data dimensions match what's expected
cat("Rows in test dataset:", nrow(test), "\n")
cat("Length of log_class:", length(log_class), "\n")
print(head(test))

# Generate predictions on the clean test data
log_preds <- predict(log_model, test, type = "response")
log_class <- ifelse(log_preds > 0.5, 1, 0) %>% as.factor()
log_class <- factor(log_class, levels = levels(test$Churn))

# Check that the lengths match
cat("Length of test$Churn:", length(test$Churn), "\n")
cat("Length of log_class:", length(log_class), "\n")

# If they still don't match, align your data
if(length(log_class) != length(test$Churn)) {
  # Only keep test observations where predictions were successfully made
  test_subset <- test[!is.na(log_preds), ]
  cat("Reduced test set to", nrow(test_subset), "rows where predictions exist\n")
  
  # Re-do predictions on matching data
  log_class <- ifelse(log_preds[!is.na(log_preds)] > 0.5, 1, 0) %>% as.factor()
  log_class <- factor(log_class, levels = levels(test_subset$Churn))
  
  # Now use test_subset for all evaluations
  log_cm <- confusionMatrix(log_class, test_subset$Churn, positive = "1")
  
  # Also update random forest predictions
  rf_model <- randomForest(Churn ~ ., data = train, ntree = 100, importance = TRUE)
  rf_preds <- predict(rf_model, test_subset)
  rf_cm <- confusionMatrix(rf_preds, test_subset$Churn, positive = "1")
  
  # Use test_subset for the rest of your code
  test <- test_subset
} else {
  # If lengths match, proceed normally
  log_cm <- confusionMatrix(log_class, test$Churn, positive = "1")
  rf_cm <- confusionMatrix(rf_preds, test$Churn, positive = "1")
}

log_roc <- roc(as.numeric(as.character(test$Churn)), as.numeric(log_preds))
plot(log_roc, main = "Logistic Regression ROC")
auc(log_roc)

# 8. Random Forest ----
set.seed(123)
rf_model <- randomForest(Churn ~ ., data = train, ntree = 100, importance = TRUE)
rf_preds <- predict(rf_model, test)
confusionMatrix(rf_preds, test$Churn, positive = "1")
rf_prob <- predict(rf_model, test, type = "prob")[,2]
rf_roc <- roc(as.numeric(as.character(test$Churn)), rf_prob)
plot(rf_roc, main = "Random Forest ROC", col = "blue")
auc(rf_roc)
importance(rf_model)
varImpPlot(rf_model, main = "Random Forest Feature Importance")

# 9. Model Comparison ----
# Compare predictions
log_cm <- confusionMatrix(log_class, test$Churn, positive = "1")
rf_cm <- confusionMatrix(rf_preds, test$Churn, positive = "1")

# Print comparison metrics
model_comparison <- data.frame(
  Model = c("Logistic Regression", "Random Forest"),
  Accuracy = c(log_cm$overall["Accuracy"], rf_cm$overall["Accuracy"]),
  Sensitivity = c(log_cm$byClass["Sensitivity"], rf_cm$byClass["Sensitivity"]),
  Specificity = c(log_cm$byClass["Specificity"], rf_cm$byClass["Specificity"]),
  F1 = c(log_cm$byClass["F1"], rf_cm$byClass["F1"])
)
print(model_comparison)

# ROC comparison
plot(log_roc, main = "ROC Curve Comparison", col = "red")
plot(rf_roc, add = TRUE, col = "blue")
legend("bottomright", legend = c(paste("Logistic Regression (AUC =", round(auc(log_roc), 3), ")"),
                                paste("Random Forest (AUC =", round(auc(rf_roc), 3), ")")),
       col = c("red", "blue"), lwd = 2)

# 10. Detailed Prediction Results ----
prediction_results <- data.frame(
  Actual_Class = test$Churn,
  LogisticRegression_Prob = log_preds,
  LogisticRegression_Class = log_class,
  RandomForest_Prob = rf_prob,
  RandomForest_Class = rf_preds,
  Model_Agreement = log_class == rf_preds
)

# View top rows of prediction results
head(prediction_results)

# Save predictions if needed
write.csv(prediction_results, "churn_predictions.csv", row.names = FALSE)

# 11. Save Models ----
saveRDS(log_model, "logistic_model.rds")
saveRDS(rf_model, "randomforest_model.rds")

