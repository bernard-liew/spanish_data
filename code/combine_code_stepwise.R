# Import packages --------------------------------------------------------------

# Helper
library(tidyverse)
library(data.table)

# ML
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(mlr3viz)
library(mlr3fselect)
library(mlr3pipelines)
library(mlr3hyperband)

# Parallel
library (future)

# Stats
library(MASS)
library(mltest)
library(Metrics)

# Import data ------------------------------------------------------------------
dat <- readRDS("output/df.RDS")

np_dat <- bind_rows(dat$df_list$np$train_imp, dat$df_list$np$test_imp) #%>% group_by(outcome) %>% sample_n(50)
ap_dat <- bind_rows(dat$df_list$ap$train_imp, dat$df_list$ap$test_imp) #%>% group_by(outcome) %>% sample_n(50)
dis_dat <- bind_rows(dat$df_list$dis$train_imp, dat$df_list$dis$test_imp) #%>% group_by(outcome) %>% sample_n(50)

# Set ML task ------------------------------------------------------------------

task_np <- TaskClassif$new (id = "neckpain", backend = np_dat , target = "outcome")
task_np$nrow
task_np$feature_names
task_np$set_col_roles("ID", roles = "name")

task_ap <- TaskClassif$new (id = "armpain", backend = ap_dat , target = "outcome")
task_ap$nrow
task_ap$feature_names
task_ap$set_col_roles("ID", roles = "name")

task_dis <- TaskClassif$new (id = "disability", backend = dis_dat , target = "outcome")
task_dis$nrow
task_dis$feature_names
task_dis$set_col_roles("ID", roles = "name")

# Set preprocessing pipes-------------------------------------------------------

poe <- po("encode", method = "one-hot")
poscale <- po("scale", param_vals = list (center = TRUE, scale = TRUE))

# Set generic params------------------------------------------------------------
create_at = function(pipe_lrn, param_set){
  auto_tuner(
    method = "random_search",
    learner = GraphLearner$new(pipe_lrn, task_type = "classif"),
    resampling = rsmp("cv", folds = 3),
    measure = msr("classif.ce"),
    search_space = param_set,
    term_evals = 200
  )
}


# Define ML models -------------------------------------------------------------
######################### Logistic regression ##################################

lrn_logreg <- lrn("classif.log_reg", id = "log", predict_type = "prob")

grln_logreg <- poscale %>>%
  lrn_logreg

logreg_gl = GraphLearner$new(grln_logreg, task_type = "classif")

######################### KNN ##################################################

lrn_kknn <- lrn("classif.kknn", id = "kknn", predict_type = "prob")

grln_kknn <- poscale %>>%
  poe %>>%
  lrn_kknn

ps_knn <- ParamSet$new(
  params = list(
    ParamInt$new("kknn.k", lower = 1, upper = 10))
  )

kknn_at = create_at(grln_kknn, ps_knn)

######################### Xgboost ##############################################

lrn_xgb <- lrn("classif.xgboost", id = "xgb", predict_type = "prob",  eta = 0.01)

grln_xgb <- poscale %>>%
  poe %>>%
  lrn_xgb

ps_xgb = ParamSet$new(
 params = list(
   ParamDbl$new("xgb.eta", lower = 0.001, upper = 0.2),
   ParamDbl$new("xgb.max_depth", lower = 1, upper = 10),
   ParamDbl$new("xgb.nrounds", lower = 50, upper = 500, tags = "budget"),
   ParamDbl$new("xgb.colsample_bytree", lower = 0.5, upper = 1),
   ParamDbl$new("xgb.colsample_bylevel", lower = 0.5, upper = 1),
   ParamDbl$new("xgb.subsample", lower = 0.5, upper = 1),
   ParamDbl$new("xgb.gamma", lower = -7, upper = 6),
   ParamDbl$new("xgb.lambda", lower = -10, upper = 10),
   ParamDbl$new("xgb.alpha", lower = -10, upper = 10)
))

ps_xgb$trafo = function(x, param_set) {
 idx_gamma = grep("gamma", names(x))
 x[[idx_gamma]] = 2^(x[[idx_gamma]])

 idx_lambda = grep("lambda", names(x))
 x[[idx_lambda]] = as.integer (2^(x[[idx_lambda]]))

 idx_alpha = grep("alpha", names(x))
 x[[idx_alpha]] = as.integer (2^(x[[idx_alpha]]))

 idx_nrounds = grep("nrounds", names(x))
 x[[idx_nrounds]] = as.integer (x[[idx_nrounds]])

 idx_depth = grep("depth", names(x))
 x[[idx_depth]] = as.integer (x[[idx_depth]])

 x
}

xgb_at = create_at(grln_xgb, ps_xgb)

######################### Lasso ################################################

lrn_lasso <- lrn("classif.glmnet", id = "lasso", predict_type = "prob")

grln_lasso <- poscale %>>%
  poe %>>%
  lrn_lasso

ps_lasso =  ParamSet$new(
  params = list(ParamDbl$new("lasso.s", lower = 0, upper = 1))
  )


lasso_at = create_at(grln_lasso, ps_lasso)

######################### Random forest ########################################

lrn_rf <- lrn("classif.ranger", id = "rf", predict_type = "prob")

grln_rf  <- poscale %>>%
  poe %>>%
  lrn_rf

ps_rf <- ParamSet$new(list(
 ParamInt$new ("rf.mtry", lower = 5, upper = 15, tags = "budget"),
 ParamDbl$new ("rf.sample.fraction", lower = 0.5, upper = 1),
 ParamInt$new ("rf.min.node.size", lower = 1, upper = 20)
))

rf_at = create_at(grln_rf, ps_rf)

######################### Nnet #################################################

lrn_net <- lrn("classif.nnet", id = "nnet", predict_type = "prob")

grln_net <- poscale %>>%
  poe %>>%
  lrn_net

ps_nnet <- ParamSet$new(list (
 ParamInt$new ("nnet.size", lower = 1, upper = 10),
 ParamDbl$new ("nnet.decay", lower = 0.1, upper = 0.5)
))

nnet_at = create_at(grln_net, ps_nnet)

######################### SVM ##################################################

lrn_svm <- lrn("classif.svm", id = "svm", type = "C-classification",
               kernel = "radial", predict_type = "prob")

grln_svm <- poscale %>>%
  poe %>>%
  lrn_svm

ps_svm <- ParamSet$new(list (
 ParamDbl$new ("svm.cost", lower = 0.1, upper = 10),
 ParamDbl$new ("svm.gamma", lower = 0, upper = 5)
))

svm_at = create_at(grln_svm, ps_svm)

# Resampling performance -------------------------------------------------------
# See ## See https://mlr3gallery.mlr-org.com/posts/2020-07-27-bikesharing-demand/

## Method 1 - quick benchmarking
### Create benchmark grid

# Ability to do all tasks at once
#
tasks <- list (task_np, task_ap, task_dis)

learners <- list (logreg_gl, kknn_at, xgb_at, lasso_at, rf_at, nnet_at, svm_at)

design <- benchmark_grid(tasks = tasks,
                        learners = learners,
                        resamplings = rsmp("holdout", ratio = 0.8))

set.seed(123456)

plan(multisession, workers = 20L)
bmr <- benchmark(design,
                  store_models = TRUE,
                  store_backends = TRUE)

measures <- list (msr("classif.auc"),
                  msr("classif.acc"),
                  msr("classif.sensitivity"),
                  msr("classif.specificity"),
                  msr("classif.precision"),
                  msr("classif.fbeta"))

resample_perf <- as.data.table (bmr$score(measures = measures)) %>%
  as.data.frame() %>%
  dplyr::select (nr, task_id, learner_id, resampling_id, iteration, matches ("classif."))

# Save -------------------------------------------------------------------------

saveRDS(resample_perf,
        "resample_perf.RDS")

saveRDS(bmr,
        "resampling_models.RDS")


# Compare against stepwise model -----------------------------------------------

split_fit_predict <- function(data_index)
{

  # recover splits
  index <- bmr$resamplings$resampling[[data_index]]$instance

  # define data
  dat <- list(np_dat, ap_dat, dis_dat)[[data_index]]
  dat_train <- dat[index$train,]
  dat_test <- dat[index$test,]
  # train models
  mod_base <- glm(outcome ~ 1 + ., family = binomial(), data = dat_train[,-1])
  mod_step <- stepAIC(mod_base)
  best_model <- mod_step$formula

  best_fitted <- mod_step$fitted.values
  best_pred <- predict(mod_step, dat_test, type = "response")

  auc <- MLmetrics::AUC (factor((best_fitted > 0.5) * 1), dat_train$outcome)
  acc <- MLmetrics::Accuracy (factor((best_fitted > 0.5) * 1), dat_train$outcome)
  sens <- MLmetrics::Sensitivity (dat_train$outcome, factor((best_fitted > 0.5) * 1))
  spec <- MLmetrics::Specificity (dat_train$outcome, factor((best_fitted > 0.5) * 1))
  fbeta <- MLmetrics::FBeta_Score (dat_train$outcome, factor((best_fitted > 0.5) * 1))

  metrics_train <- data.frame (
    auc = auc,
    acc = acc,
    sens = sens,
    spec = spec,
    fbeta = fbeta
  )


  auc <- MLmetrics::AUC (factor((best_pred > 0.5) * 1), dat_test$outcome)
  acc <- MLmetrics::Accuracy (factor((best_pred > 0.5) * 1), dat_test$outcome)
  sens <- MLmetrics::Sensitivity (dat_test$outcome, factor((best_pred > 0.5) * 1))
  spec <- MLmetrics::Specificity (dat_test$outcome, factor((best_pred > 0.5) * 1))
  fbeta <- MLmetrics::FBeta_Score (dat_test$outcome, factor((best_pred > 0.5) * 1))

  metrics_test <- data.frame (
    auc = auc,
    acc = acc,
    sens = sens,
    spec = spec,
    fbeta = fbeta
  )

  return(list(best_model = best_model,
              metrics_train = metrics_train,
              metrics_test = metrics_test))

}

res_step <- lapply(1:3, split_fit_predict)
names(res_step) <- c("np", "ap", "dis")

# extract best models
lapply(res_step, "[[", "best_model")

# extract metrics
lapply(res_step, "[[", "metrics_train")
lapply(res_step, "[[", "metrics_test")

saveRDS(res_step, file="stepwise_performance.RDS")
