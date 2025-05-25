# ================================================
# EDA + Logistic Regression + Random Forest (R)
# ================================================

# 1. Setup ----
library(tidyverse)  # For data manipulation and visualization
library(GGally)     # For advanced plotting
library(DataExplorer) # For automated EDA
library(corrplot)   # For correlation plots
library(skimr)      # For data summary
library(caret)      # For model training and evaluation
library(randomForest) # For random forest models
library(pROC)       # For ROC curve analysis
library(gridExtra)  

# Load dataset
df <- read.csv("customer_churn_dataset-testing-master.csv")

# 2. Data Overview ----
glimpse(df)
dim(df)
head(df)

# 3. Data Preprocessing ----
df$Gender <- as.factor(df$Gender)
df$Subscription.Type <- as.factor(df$Subscription.Type)
df$Contract.Length <- as.factor(df$Contract.Length)
df$Churn <- as.factor(df$Churn)

# Drop CustomerID
df <- df %>% select(-CustomerID)

# Check for missing values
sum(is.na(df))

# Check for duplicates
df <- df %>% distinct()

# 4. Univariate Analysis ----
skimr::skim(df)

# 5. Bivariate & Multivariate Analysis ----
corrplot(cor(df %>% select_if(is.numeric)), method = "color", tl.cex = 0.8)

# 6. Visualization ----
df %>% 
  ggplot(aes(x = Subscription.Type, fill = Churn)) +
  geom_bar(position = "fill") +
  labs(title = "Churn Rate by Subscription Type")

# KDE Plot
df %>%
  ggplot(aes(x = Total.Spend, fill = Churn)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot: Total Spend by Churn")

# 7. Split Data ----
set.seed(123)
trainIndex <- createDataPartition(df$Churn, p = 0.8, list = FALSE)
train <- df[trainIndex, ]
test <- df[-trainIndex, ]

# 8. Logistic Regression ----
log_model <- glm(Churn ~ ., data = train, family = "binomial")
log_preds <- predict(log_model, test, type = "response")
log_class <- ifelse(log_preds > 0.5, 1, 0) %>% as.factor()
confusionMatrix(log_class, test$Churn, positive = "1")
log_roc <- roc(as.numeric(as.character(test$Churn)), as.numeric(log_preds))
plot(log_roc, main = "Logistic Regression ROC")
auc(log_roc)

# 9. Random Forest ----
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


# 12. Loading Saved Models and Using for Prediction ----


# Prepare data for prediction (test data already exists)
new_data <- test  

# Make predictions with both models
log_preds <- predict(log_model, new_data, type = "response")
log_class <- ifelse(log_preds > 0.5, 1, 0) %>% as.factor()

rf_prob <- predict(rf_model, new_data, type = "prob")[,2]
rf_preds <- predict(rf_model, new_data)

# Compare predictions
log_cm <- confusionMatrix(log_class, new_data$Churn, positive = "1")
rf_cm <- confusionMatrix(rf_preds, new_data$Churn, positive = "1")

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
log_roc_new <- roc(as.numeric(as.character(new_data$Churn)), as.numeric(log_preds))
rf_roc_new <- roc(as.numeric(as.character(new_data$Churn)), rf_prob)

plot(log_roc_new, main = "ROC Curve Comparison on Test Data", col = "red")
plot(rf_roc_new, add = TRUE, col = "blue")
legend("bottomright", legend = c(paste("Logistic Regression (AUC =", round(auc(log_roc_new), 3), ")"),
                              paste("Random Forest (AUC =", round(auc(rf_roc_new), 3), ")")),
     col = c("red", "blue"), lwd = 2)

# For detailed prediction results
prediction_results <- data.frame(
  Actual_Class = new_data$Churn,
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