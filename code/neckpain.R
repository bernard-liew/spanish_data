#' ---
#' title: "2-prelim_model"
#' author: "Bernard"
#' date: "2021-03-31"
#' output: workflowr::wflow_html
#' editor_options:
#'   chunk_output_type: console
#' ---
#' 
#' # Introduction
#' 
## -----------------------------------------------------------------------------------------------------------------------
# Helper
library (tidyverse)

# ML
library (mlr3)
library (mlr3learners)
library (mlr3tuning)
library (mlr3viz)
library (mlr3fselect)
library (mlr3pipelines)
library (mlr3hyperband)

set.seed(7832)
lgr::get_logger("mlr3")$set_threshold("warn")
lgr::get_logger("bbotk")$set_threshold("warn")

#' 
#' # Load data
#' 
## -----------------------------------------------------------------------------------------------------------------------
dat <- readRDS("output/df.RDS") 

train <- dat$df_list$np$train_imp
test <- dat$df_list$np$test_imp

comb <- bind_rows(train, test)
train_id <- 1: nrow (train)
test_id <- (nrow (train) + 1): nrow (comb)


#' 
#' # Set task
#' 
## -----------------------------------------------------------------------------------------------------------------------
# Set training task
task<- TaskClassif$new (id = "neckpain", backend = comb, target = "outcome")
task$nrow
task$feature_names
task$set_col_roles("ID", roles = "name")

# # Set test task 
# task_tr <- TaskClassif$new (id = "neckpain", backend = test, target = "imp_np")
# task_tr$set_col_roles("ID", roles = "name")

# Set pre proc sets
poe <- po("encode", method = "one-hot")
poe$train(list(task))[[1]]$data()

poscale <- po("scale", param_vals = list (center = TRUE, scale = TRUE))
poscale$train(list(task))[[1]]$data()



#' 
#' # Set tuning
#' 
## -----------------------------------------------------------------------------------------------------------------------
evals <- trm("none")
measure <-  msr("classif.auc")
measures <- list (msr("classif.auc"), 
                  msr("classif.acc"),
                  msr("classif.tpr"),
                  msr("classif.fpr"),
                  msr("classif.fnr"),
                  msr("classif.tnr"))
# Set resample
cv_inner <- rsmp("cv", folds = 5)
cv_outer <- rsmp("cv", folds = 3)

#' 
#' # Set logistic regression model
#' 
#' 
## -----------------------------------------------------------------------------------------------------------------------
# Set learner with type proability
lrn_logreg <- lrn("classif.log_reg", id = "log", predict_type = "prob")

# Graph with factor encoding and scaling
grln_logreg <- poscale %>>%
  #poe %>>%
  lrn_logreg 

plot (grln_logreg)
grln_logreg_lnr <- GraphLearner$new(grln_logreg)

# Set autotuner
at_grln_logreg <-  AutoFSelector$new(
  learner = grln_logreg_lnr, 
  resampling = cv_inner, 
  measure = measure,
  terminator = trm("combo"),
  fselect = fs("sequential", strategy = "sbs"),
  store_models = TRUE)

# Runs the outer loop sequentially and the inner loop in parallel
future::plan(list("sequential", "multisession"))

# Nested resampling for internal validation
rr_logreg <- resample(task, 
                    at_grln_logreg, 
                    cv_outer, 
                    store_models = TRUE)

future:::ClusterRegistry("stop")

future::plan("multisession")
# Train learner
at_grln_logreg$train (task, row_ids = train_id)
as.data.table (at_grln_logreg$archive)
at_grln_logreg$fselect_result

future:::ClusterRegistry("stop")
# Predict learner
# prediction = at_grln_logreg$predict(task, row_ids = test_id)
# autoplot(prediction, type = "roc")
# prediction$score (measures)



#' 
#' # Set kknn model
#' 
#' 
## -----------------------------------------------------------------------------------------------------------------------
# Set learner with type proability
lrn_kknn <- lrn("classif.kknn", id = "kknn", predict_type = "prob")

# Graph with factor encoding and scaling
grln_kknn <- poscale %>>%
  poe %>>%
  lrn_kknn 

plot (grln_kknn)
grln_kknn_lnr <- GraphLearner$new(grln_kknn)

# Tuning
grln_kknn_lnr$param_set$values$kknn.k <-  to_tune(1, 10)
#grln_kknn_lnr$param_set$values$threshold.thresholds <-  to_tune(p_dbl (0,1))

# Set autotuner
at_grln_kknn <-  AutoTuner$new(
  learner = grln_kknn_lnr,
  resampling = cv_inner,
  measure = measure,
  terminator = evals,
  tuner = tnr("grid_search", resolution = 10),
  store_models = TRUE
)

# Runs the outer loop sequentially and the inner loop in parallel
future::plan(list("sequential", "multisession"))

# Nested resampling for internal validation
rr_kknn <- resample(task, 
                    at_grln_kknn, 
                    cv_outer, 
                    store_models = TRUE)
future:::ClusterRegistry("stop")

future::plan("multisession")
# Train learner
at_grln_kknn$train (task, row_ids = train_id)
at_grln_kknn$archive
at_grln_kknn$tuning_result

future:::ClusterRegistry("stop")
#grln_kknn_lnr$param_set$values <- at_grln_kknn$tuning_instance$result_learner_param_vals

# Predict learner
# prediction = at_grln_kknn$predict(task, row_ids = test_id)
# prediction$score (measures)
# autoplot(prediction, type = "roc")


#' 
#' # Set xgboost
#' 
## -----------------------------------------------------------------------------------------------------------------------
lrn_xgb <- lrn("classif.xgboost", id = "xgb", predict_type = "prob",  eta = 0.01)

grln_xgb <- poscale %>>%
  poe %>>%
  lrn_xgb

plot (grln_xgb)

grln_xgb_lnr <- GraphLearner$new(grln_xgb)
grln_xgb_lnr$param_set 

ps_xgb = ParamSet$new(
    params = list(
        ParamDbl$new("xgb.eta", lower = 0.001, upper = 0.2),
        ParamDbl$new("xgb.max_depth", lower = 1, upper = 20),
        ParamDbl$new("xgb.nrounds", lower = 100, upper = 5000, tags = "budget"),
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


bind_rows(generate_design_grid(ps_xgb, 3)$transpose())


at_grln_xgb <-  AutoTuner$new (
  learner = grln_xgb_lnr,
  resampling = cv_inner,
  measure = measure,
  search_space = ps_xgb,
  terminator = evals,
  tuner = tnr("hyperband", eta = 5),
  store_models = TRUE
)
# Runs the outer loop sequentially and the inner loop in parallel
future::plan(list("sequential", "multisession"))

# Nested resampling for internal validation
rr_xgb <- resample(task, 
                    at_grln_xgb, 
                    cv_outer, 
                    store_models = TRUE)

future::plan("multisession")
# test learner
at_grln_xgb$train (task, row_ids = train_id)
at_grln_xgb$archive
at_grln_xgb$tuning_result

# prediction = at_grln_xgb$predict(task, row_ids = test_id)
# prediction$score (measures)
# autoplot(prediction, type = "roc")


#' 
#' # Set lasso
#' 
## -----------------------------------------------------------------------------------------------------------------------
lrn_lasso <- lrn("classif.glmnet", id = "lasso", predict_type = "prob")

grln_lasso <- poscale %>>%
  poe %>>%
  lrn_lasso 

plot (grln_lasso)

grln_lasso_lnr <- GraphLearner$new(grln_lasso)
grln_lasso_lnr$param_set 

grln_lasso_lnr$param_set$values$lasso.s  <-  to_tune(0, 1)

at_grln_lasso <-  AutoTuner$new (
  learner = grln_lasso_lnr,
  resampling = cv_inner,
  measure = measure,
  terminator = evals,
  tuner = tnr("grid_search", resolution = 100),
  store_models = TRUE
)
# Runs the outer loop sequentially and the inner loop in parallel
future::plan(list("sequential", "multisession"))

# Nested resampling for internal validation
rr_lasso <- resample(task, 
                    at_grln_lasso, 
                    cv_outer, 
                    store_models = TRUE)

future::plan("multisession")
# test learner
at_grln_lasso$train (task, row_ids = train_id)
at_grln_lasso$archive
at_grln_lasso$tuning_result

# prediction = at_grln_lasso$predict(task, row_ids = test_id)
# prediction$score (measures)
# autoplot(prediction, type = "roc")

#' 
#' # Set random forest
#' 
## -----------------------------------------------------------------------------------------------------------------------
lrn_rf <- lrn("classif.ranger", id = "rf", predict_type = "prob")

grln_rf  <- poscale %>>%
  poe %>>%
  lrn_rf 

plot (grln_rf)

grln_rf_lnr <- GraphLearner$new(grln_rf)
grln_rf_lnr$param_set 

ps_rf <- ParamSet$new(list (
  ParamInt$new ("rf.mtry", lower = 5, upper = 15),
  ParamInt$new ("rf.min.node.size", lower = 1, upper = 20),
  ParamDbl$new ("rf.sample.fraction", lower = 0.5, upper = 1)
))

bind_rows(generate_design_grid(ps_rf, 5)$transpose())


at_grln_rf <- AutoTuner$new (
  learner = grln_rf_lnr,
  resampling = cv_inner,
  measure = measure,
  terminator = evals,
  search_space = ps_rf,
  tuner = tnr("grid_search", resolution = 5),
  store_models = TRUE
)
# Runs the outer loop sequentially and the inner loop in parallel
future::plan(list("sequential", "multisession"))

# Nested resampling for internal validation
rr_rf <- resample(task, 
                    at_grln_rf, 
                    cv_outer, 
                    store_models = TRUE)

future::plan("multisession")
# test learner
at_grln_rf$train (task, row_ids = train_id)
at_grln_rf$archive
at_grln_rf$tuning_result

# prediction = at_grln_rf$predict(task, row_ids = test_id)
# prediction$score (measures)
# autoplot(prediction, type = "roc")

#' 
#' # Set neural net
#' 
## -----------------------------------------------------------------------------------------------------------------------
lrn_net <- lrn("classif.nnet", id = "nnet", predict_type = "prob")

grln_net <- poscale %>>%
  poe %>>%
  lrn_net 

plot (grln_net)

grln_net_lnr <- GraphLearner$new(grln_net)
grln_net_lnr$param_set 

ps_net <- ParamSet$new(list (
  ParamInt$new ("nnet.size", lower = 1, upper = 10),
  ParamDbl$new ("nnet.decay", lower = 0.1, upper = 0.5)
))

bind_rows(generate_design_grid(ps_net, 5)$transpose())


at_grln_net <- AutoTuner$new (
  learner = grln_net_lnr,
  resampling = cv_inner,
  measure = measure,
  terminator = evals,
  search_space = ps_net,
  tuner = tnr("grid_search", resolution = 10),
  store_models = TRUE
)
# Runs the outer loop sequentially and the inner loop in parallel
future::plan(list("sequential", "multisession"))

# Nested resampling for internal validation
rr_net <- resample(task, 
                    at_grln_net, 
                    cv_outer, 
                    store_models = TRUE)

future::plan("multisession")
# test learner
at_grln_net$train (task, row_ids = train_id)
at_grln_net$archive
at_grln_net$tuning_result

# prediction = at_grln_net$predict(task, row_ids = test_id)
# prediction$score (measures)
# autoplot(prediction, type = "roc")

#' 
#' # Set support vector machine
#' 
## -----------------------------------------------------------------------------------------------------------------------
lrn_svm <- lrn("classif.svm", id = "svm", type = "C-classification", kernel = "radial", predict_type = "prob")

grln_svm <- poscale %>>%
  poe %>>%
  lrn_svm 

plot (grln_svm)

grln_svm_lnr <- GraphLearner$new(grln_svm)
grln_svm_lnr$param_set 

ps_svm <- ParamSet$new(list (
  ParamDbl$new ("svm.cost", lower = 0.1, upper = 10),
  ParamDbl$new ("svm.gamma", lower = 0, upper = 5)
))

bind_rows(generate_design_grid(ps_svm, 5)$transpose())


at_grln_svm <- AutoTuner$new (
  learner = grln_svm_lnr,
  resampling = resampling,
  measure = measure,
  terminator = evals,
  search_space = ps_svm,
  tuner = tnr("grid_search", resolution = 10),
  store_models = TRUE
)
# Runs the outer loop sequentially and the inner loop in parallel
future::plan(list("sequential", "multisession"))

# Nested resampling for internal validation
rr_svm <- resample(task, 
                    at_grln_svm, 
                    cv_outer, 
                    store_models = TRUE)

future::plan("multisession")
# test learner
at_grln_svm$train (task, row_ids = train_id)
at_grln_svm$archive
at_grln_svm$tuning_result

# prediction = at_grln_svm$predict(task, row_ids = test_id)
# prediction$score (measures)
# autoplot(prediction, type = "roc")

#' 
#' 
#' # Save files
## -----------------------------------------------------------------------------------------------------------------------
rsmp_list <- list (rr_logreg = rr_logreg,
                   rr_kknn = rr_kkn,
                   rr_xgb = rr_xgb,
                   rr_lasso = rr_lasso,
                   rr_rf = rr_rf,
                   rr_net = rr_net,
                   rr_svm = rr_svm)

model_list <- list (at_grln_logreg = at_grln_logreg,
                   at_grln_kknn = at_grln_kkn,
                   at_grln_xgb = at_grln_xgb,
                   at_grln_lasso = at_grln_lasso,
                   at_grln_rf = at_grln_rf,
                   at_grln_net = at_grln_net,
                   at_grln_svm = at_grln_svm)

saveRDS (list (rsmp_list = rsmp_list,
               model_list = model_list),
         "output/np_result.RDS")


#' 
#' 
