# 12. Loading Saved Models and Making Predictions ----

# Load the saved models
log_model <- readRDS("logistic_model.rds")
rf_model <- readRDS("randomforest_model.rds")

new_data <- test

# Make predictions with both models
log_preds <- predict(log_model, new_data, type = "response")
log_class <- ifelse(log_preds > 0.5, "Yes", "No") %>% as.factor()

rf_preds <- predict(rf_model, new_data)
rf_prob <- predict(rf_model, new_data, type = "prob")[,2]


# Compare predictions if you have actual labels
if ("Churn" %in% colnames(new_data)) {
  log_cm <- confusionMatrix(log_class, new_data$Churn, positive = "Yes")
  rf_cm <- confusionMatrix(rf_preds, new_data$Churn, positive = "Yes")
  
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
  log_roc <- roc(new_data$Churn == "Yes", as.numeric(log_preds))
  rf_roc <- roc(new_data$Churn == "Yes", rf_prob)
  
  plot(log_roc, main = "ROC Curve Comparison on New Data", col = "red")
  plot(rf_roc, add = TRUE, col = "blue")
  legend("bottomright", legend = c(paste("Logistic Regression
   (AUC =", round(auc(log_roc), 3), ")"),
                                   paste("Random Forest (AUC =", 
                                    round(auc(rf_roc), 3), ")")),
         col = c("red", "blue"), lwd = 2)
}

# For predictions on unlabeled data
prediction_results <- data.frame(
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