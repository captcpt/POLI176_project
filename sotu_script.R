## Load packages
library(tidyverse)
library(tidymodels)
library(tokenizers)
library(quanteda)
library(quanteda.textplots)
library(quanteda.textmodels)
library(stm)
library(seededlda)
library(devtools)
library(irr)
library(here)
library(vistime)
library(ggvis)
library(caret)
library(klaR)
library(randomForest)
### Read in data
metadata <- read_csv("data/SOTU_WithText.csv")

metadata <- metadata |>
  filter(party %in% c("Democratic", "Republican"))

# Create a corpus of the state of the union speeches
corpus_sotu <- corpus(metadata, text_field = "text")

## Pre-processing
toks <- tokens(corpus_sotu, remove_punct = TRUE, remove_numbers=TRUE)
toks <- tokens_wordstem(toks)
toks <- tokens_select(toks,  stopwords("en"), selection = "remove")
dfm <- dfm(toks)

# We can trim the dfm of rare words
dfm_trimmed <- dfm_trim(dfm, min_docfreq = 0.05, docfreq_type = "prop")

# Creating a list of gender/equality words
interest_words <- c('gender','woman','female','equality','feminism','equity','equality','rights','norms')

### EDA

#### Summary Statistics
metadata |>
  mutate(num_words = str_count(text,'\\s+'),
         num_interest = str_count(text, paste0("\\b", interest_words, "\\b", collapse = "|"))) |>
  summary(metadata)

#### Number of words per speech over time
metadata |>
  mutate(num_words = str_count(text,'\\s+')) |>
  ggplot(aes(x=year,y=num_words)) +
  geom_line() +
  labs(title="Number of words per SOTU address over time",
       x="Year",
       y="Number of words")

#### Number of interest words over time
metadata |>
  mutate(num_interest = str_count(text, paste0("\\b", interest_words, "\\b", collapse = "|"))) |>
  ggplot(aes(x=year,y=num_interest)) +
  geom_line() +
  labs(title="Number of INTEREST words per SOTU address over time",
       x="Year",
       y="Number of interest words")

#### Timeline Chart
metadata |>
  group_by(president) |>
  mutate(start = as.Date(paste0(min(year), "-01-01")), 
         end = as.Date(paste0(max(year), "-12-31")),
         event = "") |>
  gg_vistime(col.event = "event", col.start = "start", col.end = "end", col.group = "party")


### Analysis
# Selecting interest words from dfm for prediction
dfm_temp <- dfm_select(dfm, interest_words)

#### Splitting Data
docvars(corpus_sotu, "id_numeric") <- 1:ndoc(corpus_sotu)
alldocs <- 1:ndoc(corpus_sotu)
training <- sample(alldocs, round(length(alldocs)*.75))
validation <- alldocs[!alldocs%in%training]

dfmat_train <- dfm_subset(dfm_temp, docvars(corpus_sotu, "id_numeric")%in%training)
dfmat_val <- dfm_subset(dfm_temp, docvars(corpus_sotu, "id_numeric")%in%validation)

# Perform cross-validation
set.seed(123)
num_folds <- 5

# Create a data frame with the predictor variables
predictors <- as.data.frame(dfmat_train)

# Add the target variable to the data frame
predictors$party <- as.factor(docvars(dfmat_train, "party"))

# Set the control parameters for cross-validation
control <- trainControl(method = "cv", number = num_folds)

# Perform cross-validation
cv_results <- train(party ~ ., data = predictors, method = "rf", trControl = control)

# Print the cross-validation results
print(cv_results)

#### Creating Naive Bayes Model
# Training the model
tmod_rf <- randomForest(party ~ ., data = predictors)  # Change method to "randomForest"
summary(tmod_rf)

##### Checking validation/precision of training set
# Validation [Train]
predict.train_rf <- predict(tmod_rf, dfmat_train)

tab_train_rf <- table(docvars(dfmat_train, "party"), predict.train_rf)
tab_train_rf

# Precision [Train]
diag(tab_train_rf)/colSums(tab_train_rf)
# recall
diag(tab_train_rf)/rowSums(tab_train_rf)

##### Checking validation/precision of testing set
# Validation [Test]
predict.val_rf <- predict(tmod_rf, newdata = dfmat_val)

tab_val_rf <- table(docvars(dfmat_val, "party"), predict.val_rf)
tab_val_rf

# Precision [Test]
diag(tab_val_rf)/colSums(tab_val_rf)
# recall
diag(tab_val_rf)/rowSums(tab_val_rf)
