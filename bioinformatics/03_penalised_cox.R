# Stratified 7:3 split of the cohort and penalised Cox regression of the prognostic candidates.
# Manuscript: Results 2.1, Methods 4.2 and 4.3.
#
# Input:
#   significantGeneExpression.csv   column 1 sample, column 2 time (years), column 3 status (1 = death),
#                                   remaining columns expression of the candidate genes
# Outputs:
#   Training_set_raw.csv, Validation_set_raw.csv   the two cohorts (also in data/signature/)
#   All_samples_with_group.csv
#   selected_genes.csv                              genes with non-zero coefficients

library(data.table)
library(glmnet)
library(survival)

set.seed(12345)

dat <- fread("significantGeneExpression.csv")
colnames(dat)[1:3] <- c("sample", "time", "status")
dat$time   <- as.numeric(dat$time)
dat$status <- as.numeric(dat$status)

# ---- split 7:3, stratified by survival status --------------------------------
train_index <- c()
for (s in unique(dat$status)) {
  idx <- which(dat$status == s)
  train_index <- c(train_index, sample(idx, round(length(idx) * 0.7)))
}
train <- dat[train_index, ]
test  <- dat[-train_index, ]

dat$Group <- ifelse(dat$sample %in% train$sample, "Train", "Validation")
write.csv(dat,   "All_samples_with_group.csv", row.names = FALSE)
write.csv(train, "Training_set_raw.csv",       row.names = FALSE)
write.csv(test,  "Validation_set_raw.csv",     row.names = FALSE)

# ---- features: standardised with the training mean and standard deviation ----
x_train <- as.data.frame(train[, 4:ncol(train)])
constant <- vapply(x_train, function(v) sd(v, na.rm = TRUE) == 0, logical(1))
x_train <- x_train[, !constant, drop = FALSE]

centre <- vapply(x_train, mean, numeric(1), na.rm = TRUE)
spread <- vapply(x_train, sd,   numeric(1), na.rm = TRUE)
x_train <- as.matrix(sweep(sweep(x_train, 2, centre, "-"), 2, spread, "/"))
y_train <- Surv(train$time, train$status)

# ---- mixing parameter: 10-fold cross-validation, repeated five times ---------
alpha_grid <- c(0.2, 0.4, 0.6, 0.8, 1.0)   # 1.0 is the LASSO
n_repeat <- 5

best_alpha <- numeric(n_repeat)
times_selected <- setNames(rep(0, ncol(x_train)), colnames(x_train))

for (r in 1:n_repeat) {
  set.seed(1000 + r)
  best <- list(cvm = Inf)
  for (a in alpha_grid) {
    cv <- cv.glmnet(x_train, y_train, family = "cox", alpha = a, nfolds = 10)
    if (min(cv$cvm) < best$cvm) {
      best <- list(alpha = a, cv = cv, cvm = min(cv$cvm))
    }
  }
  best_alpha[r] <- best$alpha
  beta <- as.matrix(coef(best$cv, s = "lambda.min"))
  chosen <- rownames(beta)[beta[, 1] != 0]
  times_selected[chosen] <- times_selected[chosen] + 1
}

alpha_star <- as.numeric(names(which.max(table(best_alpha))))

# ---- final fit at the chosen mixing parameter, penalty at lambda.min ---------
set.seed(2024)
cv_final <- cv.glmnet(x_train, y_train, family = "cox", alpha = alpha_star, nfolds = 10)
beta <- as.matrix(coef(cv_final, s = "lambda.min"))
nonzero <- rownames(beta)[beta[, 1] != 0]

# keep genes selected in at least 60% of the repeats
stable <- intersect(nonzero, names(which(times_selected >= ceiling(n_repeat * 0.6))))

cat("alpha =", alpha_star, " lambda.min =", cv_final$lambda.min, "\n")
cat("non-zero genes:", length(nonzero), " stable genes:", length(stable), "\n")

write.csv(data.frame(Gene = stable, Coefficient = as.numeric(beta[stable, 1])),
          "selected_genes.csv", row.names = FALSE)
