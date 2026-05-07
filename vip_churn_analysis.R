# ============================================================
# VIP Customer Churn Prediction — SVM-Based Classification
# Course: Business Data Analytics II
# Institution: Sookmyung Women's University
# Author: Rayun Sa
# ============================================================


#### STEP 0: Load Data ####

library(dplyr)
library(readr)

bank <- read_csv("bank.csv")


#### STEP 1: Data Preprocessing ####

# 1) Remove identifier variables
bank$RowNumber  <- NULL
bank$CustomerId <- NULL
bank$Surname    <- NULL

# 2) Convert target variable to factor
glimpse(bank)
bank$Exited <- as.factor(bank$Exited)

# 3) Define VIP customers (Platinum + Diamond card holders)
vip <- bank %>%
  filter(`Card Type` %in% c("DIAMOND", "PLATINUM"))

# VIP ratio among all customers
vip_ratio <- nrow(vip) / nrow(bank)
vip_ratio
# Result: 0.5002 (50.02%)

# VIP churn rate
vip_churn_rate <- sum(vip$Exited == 1) / nrow(vip)
vip_churn_rate
# Result: 0.2107 (21.07%)

# Overall churn rate comparison
total_churn_rate <- sum(bank$Exited == 1) / nrow(bank)
total_churn_rate
# Overall: 0.2038 / VIP: 0.2107

# VIP share of total churned customers
1053.921 / 2038
# Result: 51.71%


#### STEP 2: Exploratory Data Analysis (EDA) ####

library(ggplot2)
install.packages("showtext")
library(showtext)

font_add_google(name = "Noto Sans KR", family = "noto")
showtext_auto()

vip <- vip %>%
  mutate(
    Geography  = as.factor(Geography),
    Gender     = as.factor(Gender),
    `Card Type` = as.factor(`Card Type`)
  )

# Age distribution
ggplot(vip, aes(x = Age)) +
  geom_histogram(bins = 30) +
  labs(title = "VIP Customer Age Distribution", x = "Age", y = "Count")

# Balance distribution (boxplot)
ggplot(vip, aes(y = Balance)) +
  geom_boxplot() +
  labs(title = "VIP Customer Balance Distribution", x = "", y = "Balance")

# Balance (log scale)
ggplot(vip, aes(y = Balance)) +
  geom_boxplot() +
  scale_y_log10() +
  labs(title = "VIP Customer Balance Distribution (log10)", x = "", y = "Balance (log10)")

# Credit score distribution
ggplot(vip, aes(x = CreditScore)) +
  geom_histogram(bins = 30) +
  labs(title = "VIP Customer Credit Score Distribution", x = "CreditScore", y = "Count")

# Geography distribution
ggplot(vip, aes(x = Geography)) +
  geom_bar() +
  labs(title = "VIP Customers by Geography", x = "Geography", y = "Count")

# Card type distribution
ggplot(vip, aes(x = `Card Type`)) +
  geom_bar() +
  labs(title = "VIP Customer Card Type Distribution", x = "Card Type", y = "Count")


#### STEP 3: Churn Pattern Analysis ####

# Age by churn status
ggplot(vip, aes(x = Exited, y = Age)) +
  geom_boxplot() +
  labs(title = "Age Distribution by Churn Status",
       x = "Churn (0=Retained, 1=Churned)", y = "Age")

# Balance by churn status (log scale)
ggplot(vip, aes(x = Exited, y = Balance)) +
  geom_boxplot() +
  scale_y_log10() +
  labs(title = "Balance Distribution by Churn Status",
       x = "Churn (0=Retained, 1=Churned)", y = "Balance (log)")

# Churn rate by active membership
ggplot(vip, aes(x = factor(IsActiveMember), fill = factor(Exited))) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Churn Rate by Active Membership",
       x = "Is Active Member", y = "Proportion", fill = "Exited")

# Churn rate by complaint history
ggplot(vip, aes(x = factor(Complain), fill = factor(Exited))) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Churn Rate by Complaint History",
       x = "Complain (0=No, 1=Yes)", y = "Proportion", fill = "Exited")

# Satisfaction score by churn status
ggplot(vip, aes(x = Exited, y = `Satisfaction Score`)) +
  geom_boxplot() +
  labs(title = "Satisfaction Score by Churn Status",
       x = "Churn (0=Retained, 1=Churned)", y = "Satisfaction Score")


#### STEP 4: Correlation Analysis ####

library(corrplot)

num_vars <- vip %>%
  select(Age, Balance, CreditScore, Tenure, `Satisfaction Score`)

cor_matrix <- cor(num_vars, use = "complete.obs")

corrplot(cor_matrix,
         method = "square",
         type   = "upper",
         tl.col = "black",
         tl.cex = 0.6,
         tl.srt = 45)


#### STEP 5: Data Preparation for SVM ####

# Convert categorical variables to factors
vip$Geography  <- as.factor(vip$Geography)
vip$Gender     <- as.factor(vip$Gender)
vip$`Card Type` <- as.factor(vip$`Card Type`)

# Dummy encoding (exclude target variable)
vip_x <- model.matrix(Exited ~ . - 1, data = vip)
vip_x <- as.data.frame(vip_x)

# Standardise features (exclude target)
vip_svm         <- as.data.frame(scale(vip_x))
vip_svm$Exited  <- vip$Exited

# Train/test split (70:30)
set.seed(123)
n         <- nrow(vip_svm)
train_idx <- sample(seq_len(n), size = floor(0.7 * n), replace = FALSE)

vip_svm_train <- vip_svm[train_idx, ]
vip_svm_test  <- vip_svm[-train_idx, ]

# Check churn rate balance across splits
table(vip_svm$Exited == "1")       / nrow(vip_svm)
table(vip_svm_train$Exited == "1") / nrow(vip_svm_train)
table(vip_svm_test$Exited == "1")  / nrow(vip_svm_test)


#### STEP 6: SVM Modelling — 4 Kernel Comparison ####

library(e1071)
library(caret)

# --- Linear Kernel ---
set.seed(333)
linear.svm <- tune.svm(Exited ~ .,
                        data   = vip_svm_train,
                        kernel = "linear",
                        cost   = c(0.02, 0.04, 0.06, 0.08, 0.1,
                                   0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 7, 10))
summary(linear.svm)
linear.svm$best.model

linear.test <- predict(linear.svm$best.model, newdata = vip_svm_test)
confusionMatrix(linear.test, vip_svm_test$Exited)

# --- Polynomial Kernel ---
set.seed(444)
poly.svm <- tune.svm(Exited ~ .,
                      data   = vip_svm_train,
                      kernel = "polynomial",
                      degree = c(2:3),
                      gamma  = c(0.01, 0.05, 0.1, 0.5),
                      coef0  = c(0, 0.5, 1),
                      cost   = c(0.1, 0.5, 1, 2, 5))
summary(poly.svm)
poly.svm$best.model

poly.test <- predict(poly.svm$best.model, newdata = vip_svm_test)
confusionMatrix(poly.test, vip_svm_test$Exited)

# --- RBF Kernel ---
set.seed(555)
rbf.svm <- tune.svm(Exited ~ .,
                     data   = vip_svm_train,
                     kernel = "radial",
                     gamma  = c(0.01, 0.05, 0.1, 0.5, 1),
                     cost   = c(0.1, 0.5, 1, 2, 5))
summary(rbf.svm)
rbf.svm$best.model

rbf.test <- predict(rbf.svm$best.model, newdata = vip_svm_test)
confusionMatrix(rbf.test, vip_svm_test$Exited)

# --- Sigmoid Kernel ---
set.seed(666)
sigmoid.svm <- tune.svm(Exited ~ .,
                         data   = vip_svm_train,
                         kernel = "sigmoid",
                         gamma  = c(0.01, 0.05, 0.1, 0.5, 1),
                         cost   = c(0.1, 0.5, 1, 2, 5))
summary(sigmoid.svm)
sigmoid.svm$best.model

sigmoid.test <- predict(sigmoid.svm$best.model, newdata = vip_svm_test)
confusionMatrix(sigmoid.test, vip_svm_test$Exited)


#### STEP 7: Final Model — Linear SVM (Refined) ####

set.seed(777)
linear.svm2 <- tune.svm(Exited ~ .,
                          data   = vip_svm_train,
                          kernel = "linear",
                          cost   = seq(0.01, 2, by = 0.05))
summary(linear.svm2)
linear.svm2$best.model

linear.test2 <- predict(linear.svm2$best.model, newdata = vip_svm_test)
confusionMatrix(linear.test2, vip_svm_test$Exited)

# Final performance summary
# Accuracy: 99.87% | Kappa: 0.9959 | Sensitivity: 0.9983 | Specificity: 1.0000
