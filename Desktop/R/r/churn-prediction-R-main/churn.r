# --------------------------------------------------
# Customer Churn Prediction and Analysis in R--Updated
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
# 1. Load Training Dataset
# --------------------------------------------------

train_df <- read.csv("churn.csv", stringsAsFactors = FALSE)

# --------------------------------------------------
# 2. Dataset Understanding
# --------------------------------------------------

# 2.2 Dataset Description
cat("\n--- Dataset Description ---\n")
cat("Training set:", nrow(train_df), "rows,", ncol(train_df), "columns\n")

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

# 3.1.1 Handle Missing Values in Training Data
train_df_clean <- train_df %>% drop_na()
cat("\n--- After removing missing values ---\n")
cat("Original training rows:", nrow(train_df), "\n")
cat("Clean training rows:", nrow(train_df_clean), "\n")
train_df <- train_df_clean

cat("\n--- Missing Values After Handling ---\n")
print(colSums(is.na(train_df)))

# 3.2 Data Type Conversion
train_df$Churn <- as.factor(train_df$Churn)

# 3.3 Duplicate and Outlier Detection
cat("\n--- Duplicate Rows ---\n")
cat("Train duplicates:", sum(duplicated(train_df)), "\n")

# Outlier Detection for numeric features
numeric_train <- train_df %>% select(where(is.numeric))
outliers <- sapply(numeric_train, function(x) sum(x %in% boxplot.stats(x)$out))
cat("\n--- Outliers per Numeric Column ---\n")
print(outliers)

# 3.4 Feature Engineering
if ("Age" %in% names(train_df)) {
  train_df$AgeGroup <- cut(train_df$Age, breaks = c(0,30,45,60,100),
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

# 7.0 Set factor levels
fixed_lvls <- c("0", "1")  # Change if your dataset has labels like "No", "Yes"
train_df$Churn <- factor(train_df$Churn, levels = fixed_lvls)

# 7.1 Logistic Regression
log_model <- glm(Churn ~ ., data = train_df, family = "binomial")
log_pred_prob <- predict(log_model, type = "response")
log_pred_class <- factor(ifelse(log_pred_prob > 0.5, fixed_lvls[2], fixed_lvls[1]), levels = fixed_lvls)

# 7.2 Random Forest
set.seed(42)
rf_model <- randomForest(Churn ~ ., data = train_df, ntree = 100, importance = TRUE)
rf_pred <- predict(rf_model)
rf_pred_prob <- predict(rf_model, type = "prob")[,2]

# --------------------------------------------------
# 8. Model Evaluation
# --------------------------------------------------

# ROC Curve - Logistic Regression
log_roc <- roc(train_df$Churn, log_pred_prob, levels = fixed_lvls)
plot(log_roc, col = "blue", main = "ROC Curve - Logistic Regression")

# ROC Curve - Random Forest
rf_roc <- roc(train_df$Churn, rf_pred_prob, levels = fixed_lvls)
plot(rf_roc, col = "red", main = "ROC Curve - Random Forest")

# Combined ROC Curve
plot(log_roc, col = "blue", main = "ROC Curve Comparison")
lines(rf_roc, col = "red")
legend("bottomright", 
       legend = c(paste("Logistic Regression (AUC =", round(auc(log_roc), 3), ")"),
                  paste("Random Forest (AUC =", round(auc(rf_roc), 3), ")")),
       col = c("blue", "red"), lwd = 2)

# Confusion Matrices
cat("\n--- Confusion Matrix - Logistic Regression ---\n")
print(confusionMatrix(log_pred_class, train_df$Churn, positive = fixed_lvls[2]))

cat("\n--- Confusion Matrix - Random Forest ---\n")
print(confusionMatrix(rf_pred, train_df$Churn, positive = fixed_lvls[2]))

# Feature Importance Plot
varImpPlot(rf_model, main = "Random Forest - Feature Importance")

# DONE ✅
