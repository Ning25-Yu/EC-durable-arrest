# Differential expression between tumour and normal endometrium (limma).
# Manuscript: Results 2.1, Methods 4.3, Figure S1.
#
# Inputs (not distributed here; see README, section "Bioinformatic part"):
#   correction.csv       gene-by-sample expression matrix (first column: gene symbol)
#   normal_samples.txt   one sample identifier per line (normal tissue)
#   cancer_samples.txt   one sample identifier per line (tumour tissue)
# Outputs:
#   DE_significant_genes.xls, DE_heatmap.pdf, DE_volcano.pdf

library(limma)
library(pheatmap)
library(ggplot2)

threshold_logFC <- 1      # absolute log2 fold change
threshold_adjP  <- 0.05   # Benjamini-Hochberg adjusted P value
max_display_genes <- 50   # genes shown on each side of the heatmap

# ---- expression matrix -------------------------------------------------------
raw_expr <- read.csv("correction.csv", header = TRUE, check.names = FALSE)
expr <- as.matrix(raw_expr[, -1])
rownames(expr) <- raw_expr[, 1]
mode(expr) <- "numeric"

expr <- avereps(expr)                  # average duplicated gene symbols
expr <- normalizeBetweenArrays(expr)   # quantile normalisation between samples

# ---- sample groups -----------------------------------------------------------
normal_ids <- read.table("normal_samples.txt", sep = "\t", header = FALSE, check.names = FALSE)[, 1]
cancer_ids <- read.table("cancer_samples.txt", sep = "\t", header = FALSE, check.names = FALSE)[, 1]

expr_normal <- expr[, as.vector(normal_ids)]
expr_cancer <- expr[, as.vector(cancer_ids)]
expr_all <- cbind(expr_normal, expr_cancer)

group <- factor(c(rep("normal", ncol(expr_normal)), rep("cancer", ncol(expr_cancer))))

# ---- linear model: cancer versus normal --------------------------------------
design <- model.matrix(~ 0 + group)
colnames(design) <- levels(group)

fit <- lmFit(expr_all, design)
contrast <- makeContrasts(Comparison = cancer - normal, levels = design)
fit <- eBayes(contrasts.fit(fit, contrast))

results <- topTable(fit, adjust.method = "fdr", number = Inf)

significant <- results[abs(results$logFC) > threshold_logFC & results$adj.P.Val < threshold_adjP, ]
significant$Regulation  <- ifelse(significant$logFC > 0, "Up regulated", "Down regulated")
significant$FoldChange  <- 2^significant$logFC
significant$Mean_normal <- rowMeans(expr_normal)[rownames(significant)]
significant$Mean_cancer <- rowMeans(expr_cancer)[rownames(significant)]

write.table(cbind(Gene = rownames(significant), significant),
            file = "DE_significant_genes.xls", sep = "\t", quote = FALSE, row.names = FALSE)

# ---- heatmap of the most strongly changed genes ------------------------------
ordered <- rownames(significant)[order(significant$logFC)]
n <- length(ordered)
if (n > 2 * max_display_genes) {
  shown <- ordered[c(1:max_display_genes, (n - max_display_genes + 1):n)]
} else {
  shown <- ordered
}

annotation <- data.frame(Group = group)
rownames(annotation) <- colnames(expr_all)

pdf("DE_heatmap.pdf", height = 8, width = 10)
pheatmap(
  expr_all[shown, ],
  annotation_col    = annotation,
  annotation_colors = list(Group = c(normal = "#8582BD", cancer = "#f5aa61")),
  color             = colorRampPalette(c("#2166AC", "white", "#B2182B"))(255),
  breaks            = seq(-2.5, 2.5, length.out = 256),
  cluster_rows      = TRUE,
  cluster_cols      = FALSE,
  show_colnames     = FALSE,
  show_rownames     = FALSE,
  scale             = "row",
  border_color      = NA
)
dev.off()

# ---- volcano plot ------------------------------------------------------------
results$Significance <- "Not significant"
results$Significance[results$adj.P.Val < threshold_adjP & results$logFC >  threshold_logFC] <- "Up regulated"
results$Significance[results$adj.P.Val < threshold_adjP & results$logFC < -threshold_logFC] <- "Down regulated"

volcano <- ggplot(results, aes(x = logFC, y = -log10(adj.P.Val), color = Significance)) +
  geom_point(size = 2.5, alpha = 0.8) +
  scale_color_manual(values = c("Down regulated" = "#1E90FF",
                                "Not significant" = "#808080",
                                "Up regulated" = "#FF4500")) +
  geom_vline(xintercept = c(-threshold_logFC, threshold_logFC), linetype = "dashed") +
  geom_hline(yintercept = -log10(threshold_adjP), linetype = "dashed") +
  labs(x = "Log2 fold change", y = "-Log10 adjusted P value") +
  theme_minimal(base_size = 14)

ggsave("DE_volcano.pdf", volcano, width = 6, height = 6)
