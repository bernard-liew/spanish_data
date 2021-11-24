# Import packages --------------------------------------------------------------

# Helper
library (tidyverse)
library (data.table)

# ML
library (mlr3)
library (mlr3learners)
library (mlr3tuning)
library (mlr3viz)
library (mlr3fselect)
library (mlr3pipelines)
library (mlr3hyperband)

# Parallel
library (future)

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

# Define peformance measures----------------------------------------------------

evals <- trm("none")
## For tuning
measure <-  msr("classif.ce")
## For performance
measures <- list (msr("classif.auc"),
                  msr("classif.acc"),
                  msr("classif.tpr"),
                  msr("classif.fpr"),
                  msr("classif.fnr"),
                  msr("classif.tnr"))

tuner <- mlr3tuning::tnr("random_search")

# Set resampling----------------------------------------------------------------
## For tuning
cv_inner <- rsmp("cv", folds = 5)
## For performance
cv_outer <- rsmp("repeated_cv", folds = 3, repeats = 10)

# Define ML models -------------------------------------------------------------

######################### Logistic regression ##################################

lrn_logreg <- lrn("classif.log_reg", id = "log", predict_type = "prob")

grln_logreg <- poscale %>>%
  lrn_logreg

grln_logreg_lnr <- GraphLearner$new(grln_logreg)

at_grln_logreg <-  AutoFSelector$new(
  learner = grln_logreg_lnr,
  resampling = cv_inner,
  measure = measure,
  terminator = trm("combo"),
  fselect = fs("sequential", strategy = "sbs"),
  store_models = TRUE)

######################### KNN ##################################################

lrn_kknn <- lrn("classif.kknn", id = "kknn", predict_type = "prob")

grln_kknn <- poscale %>>%
  poe %>>%
  lrn_kknn

grln_kknn_lnr <- GraphLearner$new(grln_kknn)

at_grln_kknn <-  AutoTuner$new(
  learner = grln_kknn_lnr,
  resampling = cv_inner,
  measure = measure,
  terminator = evals,
  tuner = tnr("grid_search", resolution = 10),
  store_models = TRUE
)

######################### Xgboost ##############################################

lrn_xgb <- lrn("classif.xgboost", id = "xgb", predict_type = "prob",  eta = 0.01)

grln_xgb <- poscale %>>%
  poe %>>%
  lrn_xgb

grln_xgb_lnr <- GraphLearner$new(grln_xgb)

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

at_grln_xgb <-  AutoTuner$new (
  learner = grln_xgb_lnr,
  resampling = cv_inner,
  measure = measure,
  search_space = ps_xgb,
  terminator = evals,
  tuner = tnr("hyperband", eta = 5),
  store_models = TRUE
)

######################### Lasso ################################################

lrn_lasso <- lrn("classif.glmnet", id = "lasso", predict_type = "prob")

grln_lasso <- poscale %>>%
  poe %>>%
  lrn_lasso


grln_lasso_lnr <- GraphLearner$new(grln_lasso)

at_grln_lasso <-  AutoTuner$new (
  learner = grln_lasso_lnr,
  resampling = cv_inner,
  measure = measure,
  terminator = evals,
  tuner = tnr("grid_search", resolution = 100),
  store_models = TRUE
)

######################### Random forest ########################################

lrn_rf <- lrn("classif.ranger", id = "rf", predict_type = "prob")

grln_rf  <- poscale %>>%
  poe %>>%
  lrn_rf

grln_rf_lnr <- GraphLearner$new(grln_rf)

ps_rf <- ParamSet$new(list (
  ParamInt$new ("rf.mtry", lower = 5, upper = 15, tags = "budget"),
  ParamDbl$new ("rf.sample.fraction", lower = 0.5, upper = 1),
  ParamInt$new ("rf.min.node.size", lower = 1, upper = 20)

))

at_grln_rf <- AutoTuner$new (
  learner = grln_rf_lnr,
  resampling = cv_inner,
  measure = measure,
  terminator = evals,
  search_space = ps_rf,
  tuner = tnr("hyperband", eta = 5),
  store_models = TRUE
)

######################### Nnet #################################################

lrn_net <- lrn("classif.nnet", id = "nnet", predict_type = "prob")

grln_net <- poscale %>>%
  poe %>>%
  lrn_net

grln_net_lnr <- GraphLearner$new(grln_net)

ps_net <- ParamSet$new(list (
  ParamInt$new ("nnet.size", lower = 1, upper = 10),
  ParamDbl$new ("nnet.decay", lower = 0.1, upper = 0.5)
))

at_grln_net <- AutoTuner$new (
  learner = grln_net_lnr,
  resampling = cv_inner,
  measure = measure,
  terminator = evals,
  search_space = ps_net,
  tuner = tnr("grid_search", resolution = 10),
  store_models = TRUE
)

######################### SVM ##################################################

lrn_svm <- lrn("classif.svm", id = "svm", type = "C-classification",
               kernel = "radial", predict_type = "prob")

grln_svm <- poscale %>>%
  poe %>>%
  lrn_svm

grln_svm_lnr <- GraphLearner$new(grln_svm)

ps_svm <- ParamSet$new(list (
  ParamDbl$new ("svm.cost", lower = 0.1, upper = 10),
  ParamDbl$new ("svm.gamma", lower = 0, upper = 5)
))

at_grln_svm <- AutoTuner$new (
  learner = grln_svm_lnr,
  resampling = cv_inner,
  measure = measure,
  terminator = evals,
  search_space = ps_svm,
  tuner = tnr("grid_search", resolution = 10),
  store_models = TRUE
)

# Resampling performance -------------------------------------------------------
# See ## See https://mlr3gallery.mlr-org.com/posts/2020-07-27-bikesharing-demand/

## Method 1 - quick benchmarking
### Create benchmark grid

# Ability to do all tasks at once
#
# tsks <- list (task_np,
#               task_ap,
#               task_dis)
#
# lrns <- list (at_grln_logreg ,
#               at_grln_kknn ,
#               at_grln_xgb ,
#               at_grln_lasso,
#               at_grln_rf,
#               at_grln_net,
#               at_grln_svm)
#
# design <- benchmark_grid(tasks = tsks,
#                          learners = lrns,
#                          resamplings = cv_outer)
#
# ###Trigger
#
# set.seed(123456)
# bmr <- benchmark(design,
#                  store_models = TRUE,
#                  store_backends = TRUE)

## Method 2 - Nested Resampling

list <- list (po(grln_kknn_lnr), # branch 1
              po(grln_xgb_lnr),
              po(grln_lasso_lnr),
              po(grln_rf_lnr),
              po(grln_net_lnr),
              po(grln_svm_lnr)) # branch 6

pipe <- ppl("branch", list)
pipe$plot()

ps = ParamSet$new(list(
  ParamInt$new("branch.selection", lower = 1L, upper = 6L),
  # knn ps
  ParamInt$new("scale.encode.kknn.kknn.k", lower = 1L, upper = 10L),
  #xgboost ps
  ParamDbl$new("scale.encode.xgb.xgb.eta", lower = 0.001, upper = 0.2),
  ParamDbl$new("scale.encode.xgb.xgb.max_depth", lower = 1, upper = 20),
  ParamDbl$new("scale.encode.xgb.xgb.nrounds", lower = 100, upper = 5000),
  ParamDbl$new("scale.encode.xgb.xgb.colsample_bytree", lower = 0.5, upper = 1),
  ParamDbl$new("scale.encode.xgb.xgb.colsample_bylevel", lower = 0.5, upper = 1),
  ParamDbl$new("scale.encode.xgb.xgb.subsample", lower = 0.5, upper = 1),
  ParamDbl$new("scale.encode.xgb.xgb.gamma", lower = -7, upper = 6),
  ParamDbl$new("scale.encode.xgb.xgb.lambda", lower = -10, upper = 10),
  ParamDbl$new("scale.encode.xgb.xgb.alpha", lower = -10, upper = 10),
  #lasso ps
  ParamDbl$new("scale.encode.lasso.lasso.s", lower = 1, upper = 100),
  #rf ps
  ParamInt$new ("scale.encode.rf.rf.mtry", lower = 5, upper = 15),
  ParamDbl$new ("scale.encode.rf.rf.sample.fraction", lower = 0.5, upper = 1),
  ParamInt$new ("scale.encode.rf.rf.min.node.size", lower = 1, upper = 20),
  #nnet ps
  ParamInt$new ("scale.encode.nnet.nnet.size", lower = 1, upper = 10),
  ParamDbl$new ("scale.encode.nnet.nnet.decay", lower = 0.1, upper = 0.5),
  # svm ps
  ParamDbl$new ("scale.encode.svm.svm.cost", lower = 0.1, upper = 10),
  ParamDbl$new ("scale.encode.svm.svm.gamma", lower = 0, upper = 5)
))

ps$trafo <- function(x, param_set) {

    idx_gamma = grep("xgb.gamma", names(x))
    x[[idx_gamma]] = 2^(x[[idx_gamma]])

    idx_lambda = grep("xgb.lambda", names(x))
    x[[idx_lambda]] = as.integer (2^(x[[idx_lambda]]))

    idx_alpha = grep("xgb.alpha", names(x))
    x[[idx_alpha]] = as.integer (2^(x[[idx_alpha]]))

    idx_nrounds = grep("xgb.nrounds", names(x))
    x[[idx_nrounds]] = as.integer (x[[idx_nrounds]])

    idx_depth = grep("xgb.max_depth", names(x))
    x[[idx_depth]] = as.integer (x[[idx_depth]])

    idx_lambda = grep("lasso.s", names(x))
    x[[idx_lambda ]] = log(x[[idx_lambda]])

    x
}

## Start complex resampling

multi_at <- AutoTuner$new(
  learner = GraphLearner$new(pipe, task_type = "classif"),
  resampling =  cv_inner,
  measure = measure,
  search_space = ps,
  term = evals,
  tuner = tuner
)

###Trigger

# Website shows this, but this does one task only
# set.seed(123456)
# multi_at_rr_np <- resample(task_np,
#                            multi_at,
#                            cv_outer,
#                            store_models = TRUE)

# 3 task in 1

design2 <- benchmark_grid(tasks = tsks,
                         learners = list (grln_logreg_lnr,
                                          multi_at),
                         resamplings = cv_outer)

set.seed(123456)
bmr2 <- benchmark(design2,
                  store_models = TRUE,
                  store_backends = TRUE)


resample_perf <- as.data.table (bmr2$score(measures = measures)) %>%
  as.data.frame() %>%
  dplyr::select (nr, task_id, learner_id, resampling_id, iteration, matches ("classif."))

# Save -------------------------------------------------------------------------

saveRDS(resample_perf,
        "resample_perf.RDS")

saveRDS(bmr2,
        "resampling_models.RDS")

