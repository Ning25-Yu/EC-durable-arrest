# Stepwise Cox regression that gives the four-gene signature, and its performance in both cohorts.
# Manuscript: Results 2.1, Equation 1, Methods 4.3, Figures S2 and S3.
#
# Inputs (data/signature/):
#   Training_set_raw.csv, Validation_set_raw.csv   column 1 sample, column 2 time (years), column 3 status
# Outputs:
#   StepCox_final_model_coefficients.csv, StepCox_PH_assumption_test.csv,
#   Training_risk_scores.csv, Validation_risk_scores.csv, Model_performance_summary.csv
# The Python script code/analysis/25_signature_check.py repeats the selection and the performance figures.

library(data.table)
library(survival)
library(MASS)
library(timeROC)

read_cohort <- function(path) {
  d <- fread(path)
  colnames(d)[1:3] <- c("sample", "time", "status")
  d$time   <- as.numeric(d$time)
  d$status <- as.numeric(d$status)
  d
}

train <- read_cohort("../data/signature/Training_set_raw.csv")
test  <- read_cohort("../data/signature/Validation_set_raw.csv")

# ---- candidate genes entering the stepwise regression ------------------------
candidates <- c(
  "PLA2G10", "FSTL3", "B4GALT1", "ASB2", "SH3GL2", "HOXB6",
  "KLRB1", "ZAP70", "FASLG", "STK11", "FBXL16", "IRF5",
  "DUSP6", "SLC39A3", "ATP6V1C2", "TRPC1", "TSC22D3", "PTX3",
  "RET", "STAT6", "CD3D", "ANXA2", "ANG", "HLA_DMB"
)
candidates <- intersect(candidates, colnames(train))

df <- train[, c("time", "status", candidates), with = FALSE]

# ---- bidirectional stepwise selection with the BIC penalty -------------------
full <- coxph(as.formula(paste("Surv(time, status) ~", paste(candidates, collapse = " + "))),
              data = df, ties = "efron")
model <- stepAIC(full, direction = "both", k = log(nrow(df)), trace = FALSE)

s <- summary(model)
coefficients <- data.frame(
  Gene    = rownames(s$coefficients),
  Coef    = s$coefficients[, "coef"],
  HR      = s$coefficients[, "exp(coef)"],
  lower95 = s$conf.int[, "lower .95"],
  upper95 = s$conf.int[, "upper .95"],
  Pvalue  = s$coefficients[, "Pr(>|z|)"],
  row.names = NULL
)
write.csv(coefficients, "StepCox_final_model_coefficients.csv", row.names = FALSE)
write.csv(as.data.frame(cox.zph(model)$table), "StepCox_PH_assumption_test.csv")

# ---- risk score and median cutoff of the training cohort ---------------------
df$risk <- predict(model, type = "lp")
cutoff <- median(df$risk)
df$group <- ifelse(df$risk > cutoff, "High", "Low")

test$risk  <- predict(model, newdata = test, type = "lp")
test$group <- ifelse(test$risk > cutoff, "High", "Low")

write.csv(df[, c("time", "status", "risk", "group")], "Training_risk_scores.csv", row.names = FALSE)
write.csv(test[, c("sample", "time", "status", "risk", "group")], "Validation_risk_scores.csv", row.names = FALSE)

# ---- concordance and time-dependent AUC at 1, 3 and 5 years ------------------
performance <- function(d, label) {
  cindex <- summary(coxph(Surv(time, status) ~ risk, data = d))$concordance[1]
  roc <- timeROC(T = d$time, delta = d$status, marker = d$risk, cause = 1, times = c(1, 3, 5), iid = TRUE)
  data.frame(Dataset = label, C_index = cindex,
             AUC_1y = roc$AUC[1], AUC_3y = roc$AUC[2], AUC_5y = roc$AUC[3])
}

write.csv(rbind(performance(df, "Training"), performance(test, "Validation")),
          "Model_performance_summary.csv", row.names = FALSE)
