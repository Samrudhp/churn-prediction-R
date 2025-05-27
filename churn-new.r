# --------------------------------------------------
# Customer Churn Prediction and Analysis in R--new
# --------------------------------------------------

# Load Required Libraries
library(tidyverse)      # For data manipulation and visualization
library(GGally)         # For advanced plotting
library(DataExplorer)   # For automated EDA
library(corrplot)       # For correlation plots
library(skimr)          # For data summary
library(caret)          # For model training and evaluation
library(randomForest)   # For random forest
library(pROC)           # For ROC curve
library(gridExtra)      # For arranging plots

# --------------------------------------------------
# 1. Load Training and Testing Datasets
# --------------------------------------------------

train_df <- read.csv("customer_churn_dataset-training-master.csv", stringsAsFactors = FALSE)
test_df  <- read.csv("customer_churn_dataset-testing-master.csv", stringsAsFactors = FALSE)

# --------------------------------------------------
# 2. Dataset Understanding
# --------------------------------------------------

# 2.2 Dataset Description
cat("\n--- Dataset Description ---\n")
cat("Training set:", nrow(train_df), "rows,", ncol(train_df), "columns\n")
cat("Testing set:", nrow(test_df), "rows,", ncol(test_df), "columns\n")

# 2.3 Data Types and Feature Overview
cat("\n--- Data Structure ---\n")
str(train_df)

# 2.4 Target Variable
cat("\n--- Target Variable Distribution (Churn) ---\n")
print(table(train_df$Churn))

# --------------------------------------------------
# 3. Data Preprocessing
# --------------------------------------------------

# 3.1 Handling Missing Values
cat("\n--- Missing Values ---\n")
print(colSums(is.na(train_df)))
print(colSums(is.na(test_df)))

# 3.1.1 Handle Missing Values in Training Data
# Option 1: Remove rows with missing values (simplest approach)
train_df_clean <- train_df %>% drop_na()
cat("\n--- After removing missing values ---\n")
cat("Original training rows:", nrow(train_df), "\n")
cat("Clean training rows:", nrow(train_df_clean), "\n")
train_df <- train_df_clean

cat("\n--- Missing Values After Handling ---\n")
print(colSums(is.na(train_df)))



# 3.2 Data Type Conversion
train_df$Churn <- as.factor(train_df$Churn)
test_df$Churn <- as.factor(test_df$Churn)

# 3.3 Duplicate and Outlier Detection
cat("\n--- Duplicate Rows ---\n")
cat("Train duplicates:", sum(duplicated(train_df)), "\n")
cat("Test duplicates:", sum(duplicated(test_df)), "\n")

# Outlier Detection for numeric features
numeric_train <- train_df %>% select(where(is.numeric))
outliers <- sapply(numeric_train, function(x) sum(x %in% boxplot.stats(x)$out))
cat("\n--- Outliers per Numeric Column ---\n")
print(outliers)

# 3.4 Feature Engineering
if ("Age" %in% names(train_df)) {
  train_df$AgeGroup <- cut(train_df$Age, breaks = c(0,30,45,60,100),
                           labels = c("Young", "Adult", "Mid-Age", "Senior"))
  test_df$AgeGroup <- cut(test_df$Age, breaks = c(0,30,45,60,100),
                          labels = c("Young", "Adult", "Mid-Age", "Senior"))
}

# --------------------------------------------------
# 4. Univariate Analysis
# --------------------------------------------------

cat("\n--- Summary Statistics ---\n")
print(skim(train_df))

# Distribution Plots
numeric_cols <- train_df %>% select(where(is.numeric))
par(mfrow = c(2, 2))
for (col in names(numeric_cols)) {
  hist(numeric_cols[[col]], main = paste("Histogram of", col), xlab = col, col = "skyblue")
}

# Boxplots
for (col in names(numeric_cols)) {
  boxplot(train_df[[col]], main = paste("Boxplot of", col), col = "lightpink")
}

# Frequency counts (categorical)
cat_cols <- train_df %>% select(where(is.character))
for (col in names(cat_cols)) {
  print(table(train_df[[col]]))
}

# --------------------------------------------------
# 5. Bivariate and Multivariate Analysis
# --------------------------------------------------

# 5.1 Correlation Matrix
cor_data <- train_df %>% select(where(is.numeric))
cor_matrix <- cor(cor_data, use = "complete.obs")
corrplot(cor_matrix, method = "color", tl.cex = 0.8)

# 5.2 Pairplots
GGally::ggpairs(cor_data)

# 5.3 Grouped Analysis
if ("Gender" %in% names(train_df)) {
  cat("\n--- Mean Tenure by Gender ---\n")
  print(train_df %>% group_by(Gender) %>% summarise(MeanTenure = mean(Tenure, na.rm = TRUE)))
}

# --------------------------------------------------
# 6. Visualization and Insights
# --------------------------------------------------

# 6.1 Key Trends - Churn Pie Chart
churn_counts <- table(train_df$Churn)
pie(churn_counts, labels = paste0(names(churn_counts), " (", round(100*churn_counts/sum(churn_counts), 1), "%)"),
    col = c("lightblue", "salmon"), main = "Churn Distribution")

# 6.2 Heatmap of Correlations
heatmap(cor_matrix, main = "Numeric Feature Correlation Heatmap", col = heat.colors(256))

# 6.3 Violin Plots
for (col in names(numeric_cols)) {
  print(ggplot(train_df, aes(x = Churn, y = .data[[col]])) +
          geom_violin(fill = "lightgreen") +
          ggtitle(paste("Violin Plot -", col)))
}

# --------------------------------------------------
# 7. Modeling – Logistic Regression and Random Forest
# --------------------------------------------------

# Remove non-numeric/categorical leakage if necessary
# Fit Logistic Regression
log_model <- glm(Churn ~ ., data = train_df, family = "binomial")
log_pred_prob <- predict(log_model, newdata = test_df, type = "response")
log_pred_class <- ifelse(log_pred_prob > 0.5, "1", "0")  # Match test_df$Churn levels

# Before confusion matrix, check and fix factor levels:
cat("\n--- Checking factor levels ---\n")
print(levels(test_df$Churn))  # Check what levels your test data has

# Ensure prediction uses same levels as test data
log_pred_class <- factor(log_pred_class, levels = levels(test_df$Churn))

# Then your confusion matrix should work:
cat("\n--- Confusion Matrix - Logistic Regression ---\n")
print(confusionMatrix(log_pred_class, test_df$Churn))

# Fit Random Forest
rf_model <- randomForest(Churn ~ ., data = train_df, ntree = 100, importance = TRUE)
rf_pred <- predict(rf_model, newdata = test_df)

# --------------------------------------------------
# 8. Model Evaluation
# --------------------------------------------------

# ROC Curve - Logistic Regression
log_roc <- roc(test_df$Churn, as.numeric(log_pred_prob))
plot(log_roc, col = "blue", main = "ROC Curve - Logistic Regression")

# ROC Curve - Random Forest
rf_pred_prob <- predict(rf_model, newdata = test_df, type = "prob")[,2]  # Get probabilities
rf_roc <- roc(test_df$Churn, rf_pred_prob)
plot(rf_roc, col = "red", main = "ROC Curve - Random Forest")

# Combined ROC Curve for comparison
plot(log_roc, col = "blue", main = "ROC Curve Comparison")
lines(rf_roc, col = "red")
legend("bottomright", 
       legend = c(paste("Logistic Regression (AUC =", round(auc(log_roc), 3), ")"),
                 paste("Random Forest (AUC =", round(auc(rf_roc), 3), ")")),
       col = c("blue", "red"), lwd = 2)

# Confusion Matrix
cat("\n--- Confusion Matrix - Logistic Regression ---\n")
print(confusionMatrix(log_pred_class, test_df$Churn))

cat("\n--- Confusion Matrix - Random Forest ---\n")
print(confusionMatrix(rf_pred, test_df$Churn))



# Random Forest Feature Importance
varImpPlot(rf_model, main = "Random Forest - Feature Importance")

# DONE ✅
