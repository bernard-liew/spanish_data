res <- readRDS("output/dis_iml.RDS")

o <- dis_test$outcome

# P value

df <- dis_train %>%
  mutate (outcome = as.numeric(outcome) - 1)

fullmodel <- glm(outcome ~., data = df, family = binomial)
nullmodel <- glm(outcome ~1, data = df, family = binomial)

scope = list(lower=formula(nullmodel ),upper=formula(fullmodel))

dis_m1 = SignifReg(fullmodel,
                  scope=scope,
                  alpha = 0.05,
                  direction = "both",
                  criterion = "p-value",
                  adjust.method = "none",
                  trace=FALSE)

p_m1 <- predict (dis_m1, type = "response", newdata  = dis_test %>% dplyr::select (-outcome))
p_m1 <- ifelse (p_m1 > 0.5, 1, 0)

res_m1 <- performance (y_pred = p_m1,
                       y_true = o)

coef_dis1 <- data.frame (variables = names (coef(dis_m1)),
                        pval = coef(dis_m1))


pred <- res$predictors

for (n in seq_along (coef_dis1$variables[-1])) {

  var <- coef_dis1$variables[-1][n]
  pred$pval[pred$variables == var] <- coef_dis1$pval[-1][n]

}

perf <- res$performance
perf[1,2:6] <- res_m1

# P val with adjustment


dis_m1 = SignifReg(fullmodel,
                  scope=scope,
                  alpha = 0.05,
                  direction = "both",
                  criterion = "p-value",
                  adjust.method = "BY",
                  trace=FALSE)

p_m1 <- predict (dis_m1, type = "response", newdata  = dis_test %>% dplyr::select (-outcome))
p_m1 <- ifelse (p_m1 > 0.5, 1, 0)

res_m1 <- performance (y_pred = p_m1,
                       y_true = o)

coef_dis1 <- data.frame (variables = names (coef(dis_m1)),
                        pvalAdj = coef(dis_m1))


temp_perf <- data.frame(Model = "pvalAdj",
                        Accuracy = res_m1[1],
                        AUC = res_m1[2],
                        Precision = res_m1[3],
                        Sensitivity = res_m1[4],
                        Specificity = res_m1[5])
perf <- perf %>%
  bind_rows(temp_perf) %>%
  mutate (Model = factor (Model,
                          levels = c("pval", "pvalAdj", "stepaic", "bestsubset",
                                     "lasso", "ncvreg", "mboost", "mars"))) %>%
  arrange (Model)


pred <- pred %>%
  left_join(coef_dis1, by = "variables") %>%
  dplyr::select (variables, pval, pvalAdj, everything(), - c(remove, keep))

res$performance <- perf
res$predictors <- pred
res$models$pvaladj <- dis_m1

saveRDS(res, "output/dis_iml.RDS")
