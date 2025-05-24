
# ================================================
# EDA + Logistic Regression + Random Forest (R)
# ================================================

# 1. Setup ----
required_packages <- c("tidyverse", "GGally", "DataExplorer", "corrplot", "skimr", "caret", "randomForest", "pROC")
install.packages(setdiff(required_packages, rownames(installed.packages())))
lapply(required_packages, library, character.only = TRUE)

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
