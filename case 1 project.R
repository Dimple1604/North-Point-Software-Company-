library(RWeka)
library(dplyr)
library(caret)
library(tidyverse)
library(ggplot2)
library(gridExtra)
library(rpart)
library(glmnet)
library(corrplot)
library(rpart.plot)
library(psych)
##exploring and preparing the data
Npoint.df<-read.csv("C:/Users/Dimple/Desktop/Data alaystics/CSDA 6010 Analytics Practicum/North-Point List.csv")


Npoint <- Npoint.df %>%
  select(-sequence_number) %>%
  mutate(
    Purchase = factor(Purchase, levels = c(1, 0), labels = c("Yes", "No")),
    Spending = ifelse(Purchase == "No", 0, Spending)
  )

Npoint
View(Npoint)
str(Npoint)
summary(Npoint)
dim(Npoint)

sum(Npoint$Spending)

#missing values 
missing_values <- colSums(is.na(Npoint))
missing_values


#Distribution of Binary Variable
table(Npoint$Purchase)
table(Npoint$source_a)
table(Npoint$source_c)
table(Npoint$source_b)
table(Npoint$source_d)
table(Npoint$source_e)
table(Npoint$source_m)
table(Npoint$source_o)
table(Npoint$source_h)
table(Npoint$source_r)
table(Npoint$source_s)
table(Npoint$source_t)
table(Npoint$source_u)
table(Npoint$source_p)
table(Npoint$source_x)
table(Npoint$source_w)
table(Npoint$US)
table(Npoint$Web_order)
table(Npoint$Gender_male)
table(Npoint$Address_is_res)

# Create a contingency table for 'Freq' and 'Purchase'
contingency_table <- table(Npoint$Purchase, Npoint$Freq)

# Create a grouped bar chart
barplot(contingency_table, beside = TRUE, legend = c("No Purchase", "Purchase"), 
        col = c("lightblue", "lightgreen"), main = "Freq vs. Purchase")

# Create a contingency table for 'source_a' and 'Purchase'
contingency_table <- table(Npoint$Purchase, Npoint$source_a)

# Create a grouped bar chart
barplot(contingency_table, beside = TRUE, legend = c("Not purchase", "Purchase"), 
        col = c("lightblue", "lightgreen"), main = "Source_A vs. Purchase")

##weborder spending range 
ggplot(Npoint) + geom_boxplot(aes(x=as.factor(Web_order), y=Spending)) + xlab("Web order")


# Select numeric variables for correlation analysis
numeric_variables <- Npoint[, c("Freq", "last_update_days_ago", "X1st_update_days_ago", "Spending")]
cor_matrix <- cor(numeric_variables)
cor_matrix

#pairplot for numerical values
library(GGally)
ggpairs(Npoint[, c(17, 18, 19, 24)],
        lower=list(continuous=wrap("points", alpha=0.25, size=0.3)))


##..............................................................................................................................................................................
set.seed(1)

## partitioning into training (40%), validation (35%), holdout (25%)
# randomly sample 40% of the row IDs for training
train.rows <- sample(rownames(Npoint), nrow(Npoint)*0.4)

# sample 35% of the row IDs into the validation set, drawing only from records
# not already in the training set
# use setdiff() to find records not already in the training set
valid.rows <- sample(setdiff(rownames(Npoint), train.rows),
                     nrow(Npoint)*0.35)

# assign the remaining 25% row IDs serve as holdout
holdout.rows <- setdiff(rownames(Npoint), union(train.rows, valid.rows))

# create the 3 data frames by collecting all columns from the appropriate rows
train.df <- Npoint[train.rows, ]
valid.df <- Npoint[valid.rows, ]
holdout.df <- Npoint[holdout.rows, ]

#Create training and validation datasets for only purchasers
Purchasers_train <- train.df[train.df$Purchase == "Yes", ]
Purchasers_validation <- valid.df[valid.df$Purchase == "Yes", ]

dim(train.df)
dim(valid.df)
dim(holdout.df)
dim(Purchasers_train)
dim(Purchasers_validation)

str(train.df)
#data preparation for logistic regression................................................................................................................................


Npoint_logistic <- Npoint.df %>%
  select(-sequence_number) %>%
  mutate(
    Spending = ifelse(Purchase == "No", 0, Spending)
  )

Npoint_logistic

set.seed(1)
train <- sample(rownames(Npoint_logistic), nrow(Npoint_logistic)*0.4)
valid<- sample(setdiff(rownames(Npoint_logistic), train),
                     nrow(Npoint_logistic)*0.35)
holdout <- setdiff(rownames(Npoint_logistic), union(train, valid))
train_lg <- Npoint_logistic[train, ]
valid_lg <- Npoint_logistic[valid, ]
holdout_lg <- Npoint_logistic[holdout, ]
dim(train_lg) 
dim(valid_lg)
dim(holdout_lg)

##............................................Fit logistic regression model................................................................................................................................................................
# Prepare predictors
library(caret)
logit_reg <- caret::train(Purchase ~ .-Spending, train_lg , 
                          method="glm", family="binomial")

summary(logit_reg)

# Predict probabilities for the validation dataset
logit.reg.pred <- predict(logit_reg, valid_lg[, -23])

# Create a confusion matrix
confusionMatrix(factor(ifelse(logit.reg.pred > 0.5, 1, 0), levels = c(1,0)), 
                factor(valid_lg$Purchase, levels = c(1,0)))


#..................................Perform forward selection logistic regression model..............................................................................................................................................
# Define step_null and step_full 
logit.null <- glm(Purchase~1, data = train_lg, family = "binomial")
logit.full <- glm(Purchase ~ . -Spending, data = train_lg, family = "binomial")


step_forward_lm <- step(logit.null, scope=list(lower=logit.null, upper=logit.full), direction = "forward")
summary(step_forward_lm)
# Generate predictions

step.pred <- predict(step_forward_lm, valid_lg[ ,-c(23)], type = "response")


# Create a confusion matrix
confusionMatrix(factor(ifelse(step.pred > 0.5, 1, 0), levels = c(1,0)), 
                factor(valid_lg$Purchase, levels = c(1,0)))


#......................................model evaluation...........................................................................................................................................................

step.train <- predict(step_forward_lm, train_lg[ ,-c(23)], type = "response")
train_classes <- ifelse(step.train > 0.5, "No", "Yes")

confusionMatrix(factor(ifelse(step.train > 0.5, 1, 0), levels = c(1,0)), 
                factor(train_lg$Purchase, levels = c(1,0)))

#....................................Fiting classification tree model ...................................................................................................

# Fit a classification tree model
tree_model <- rpart(Purchase ~ . - Spending, data = train.df, method = "class")
tree_model

rpart.plot(tree_model, extra=1, fallen.leaves=FALSE)
# Make predictions on the validation data
tree_predictions <- predict(tree_model, newdata = valid.df, type = "class")
tree_predictions

# Calculate accuracy
tree_accuracy <- mean(tree_predictions == valid.df$Purchase)
tree_accuracy

conf_matrix_naive <- confusionMatrix(factor(tree_predictions, levels = c("Yes", "No")), 
                                     factor(valid.df$Purchase, levels = c("Yes", "No")),
                                     positive = "Yes")
conf_matrix_naive

#................................................purchase data.............................................................................................................................................................................................................

#Create training and validation datasets for only purchasers
Purchasers_train <- train.df[train.df$Purchase == "Yes", ]
Purchasers_validation <- valid.df[valid.df$Purchase == "Yes", ]


dim(Purchasers_train)
dim(Purchasers_validation)

#................................................linear regression model .......................................................................................................................................................................................................

# Histogram of Spending 
hist(Purchasers_train$Spending, main = "Histogram of Spending", xlab = "Spending")


# Exclude the 'Purchase' variable and fit the linear regression model
model_lm <- lm(Spending ~ .,data = Purchasers_train[,c(-23)])
model_lm
summary(model_lm)

lm_predictions <- predict(model_lm, newdata = Purchasers_validation[,c(-23)])
lm_predictions

forecast::accuracy(lm_predictions, Purchasers_validation$Spending)


#RMSE performance measure 
caret::RMSE(lm_predictions, Purchasers_validation$Spending)

#...........................................forward stepwise regression model...............................................................................................................................................................................................................................

# Define step_null and step_full 
step_null <- lm(Spending ~ 1, data = Purchasers_train[, c(-19, -23)])
step_full <- lm(Spending ~ ., data = Purchasers_train[, c(-19, -23)])

# Perform forward selection..............................................................................................................................................
forward_lm <- step(step_null, scope = list(lower = step_null, upper = step_full), direction = "forward")

summary(forward_lm)
# Generate predictions
pred_forward_lm <- predict(forward_lm, newdata = Purchasers_validation[, -c(19, 23)])

forecast::accuracy(pred_forward_lm, Purchasers_validation$Spending)

# Calculate RMSE
caret::RMSE(pred_forward_lm, Purchasers_validation$Spending)
#...............................................model evaluation................................................................................................................................
train_forward_lm <- predict(forward_lm, newdata = Purchasers_train[, -c(19, 23)])

forecast::accuracy(train_forward_lm, Purchasers_train$Spending)

# Calculate RMSE
caret::RMSE(train_forward_lm, Purchasers_train$Spending)

#..............................................Regression tree......................................................................................................................................................................................................

# Fit regression tree model to predict spending value
tree_model_reg <- rpart(Spending ~ ., data = Purchasers_train[,c(-23)])

# Print summary of regression tree model
tree_model_reg

# Plot the regression tree
rpart.plot(tree_model_reg)

Regtree_predict <- predict(tree_model_reg, newdata = Purchasers_validation[, -c(23, 24)])
summary(Regtree_predict)


summary(Purchasers_validation$Spending)
cor(Regtree_predict, Purchasers_validation$Spending)
forecast::accuracy(Regtree_predict, Purchasers_validation$Spending)

#...................................Improving Model Performance M5P model........................................................................................................
improve_model <- M5P(Spending ~ ., data = Purchasers_train[,c(-23)])
improve_model

summary(improve_model)

improve_predict<-predict(improve_model, Purchasers_validation[ ,c(-23,-24)])
summary(improve_predict)

cor(improve_predict, Purchasers_validation$Spending)
MAE(Purchasers_validation$Spending, improve_predict)

forecast::accuracy(improve_predict, Purchasers_validation$Spending)


#data..........................................................................................................................
# a add a new column with the predicted probability ...................................................................................................................................................................................
holdout.df$Predicted_Probability <- predict(step_forward_lm, holdout.df, type = "response")
head(holdout.df)


#b Add a column to the predicted Spending .............................................................................................................................................................................
holdout.df$Predicted_Spending <- predict(forward_lm, holdout.df)
head(holdout.df)

## original purchase rate
original.purchase.rate <- 0.1065

#c Add a column for adjusted probability of purchase...........................................................................................................................................................................................

holdout.df$Adjusted.Purchase.Probability <- holdout.df$Predicted_Probability * original.purchase.rate
head(holdout.df)

#d Add a column for Expected Spending..........................................................................................................................................................................................

holdout.df$Expected_Spending <- holdout.df$Adjusted.Purchase.Probability * holdout.df$Predicted_Spending
head(holdout.df)

#e.	Plot the cumulative gain chart................................................................................................................................................................................
library(ggplot2)
library(gridExtra)
library(gains)

# Compute gains
Spending <- holdout.df$Spending
gain <- gains(Spending, holdout.df$Expected_Spending)


# Cumulative gains chart
df_cumulative <- data.frame(
  ncases = c(0, gain$cume.obs),
  cumSpending = c(0, gain$cume.pct.of.total * sum(holdout.df$Spending))
)
df_cumulative
g1 <- ggplot(df_cumulative, aes(x = ncases, y = cumSpending)) +
  geom_line() +
  geom_line(data = data.frame(ncases = c(0, nrow(holdout.df)), cumSpending = c(0, sum(holdout.df$Spending))),
            color = "gray", linetype = 2) + # adds baseline
  labs(x = "percentage of records targeted ", y = "Cumulative expected Spending", title = "Cumulative Gains  expected spending") +
  scale_y_continuous(labels = scales::comma)
g1

# Decile-wise lift chart
df_decile <- data.frame(
  percentile = gain$depth,
  meanResponse = gain$mean.resp / mean(Spending)
)
df_decile
g2 <- ggplot(df_decile, aes(x = percentile, y = meanResponse)) +
  geom_bar(stat = "identity") +
  labs(x = "Percentile", y = "Decile mean / global mean", title = "Decile-wise Lift Chart")
g2
# Display the plots
grid.arrange(g1, g2, ncol = 2)

