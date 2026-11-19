# Volcano plot of the univariate Cox screen of the immune-related differentially expressed genes.
# Manuscript: Results 2.1, Figure S2.
#
# Input:
#   allCoxResults.csv   one row per gene with the columns Gene, coef, HR and pvalue
#                       (univariate Cox regression of overall survival on the expression of the gene)
# Output:
#   Cox_volcano.pdf
# Genes with P < 0.05 are the prognostic candidates passed to the penalised regression.

library(ggplot2)
library(ggrepel)
library(dplyr)

p_threshold <- 0.05

cox <- read.csv("allCoxResults.csv", stringsAsFactors = FALSE)

cox <- cox %>%
  mutate(
    negLogP = -log10(pvalue),
    group = case_when(
      HR > 1 & pvalue < p_threshold ~ "HR > 1, P < 0.05",
      HR < 1 & pvalue < p_threshold ~ "HR < 1, P < 0.05",
      TRUE ~ "Not significant"
    ),
    coef_plot = pmax(pmin(coef, 3), -3)   # clip extreme coefficients for display
  )

top_genes <- cox %>%
  filter(pvalue < 0.001) %>%
  arrange(pvalue) %>%
  head(10)

volcano <- ggplot(cox, aes(x = coef_plot, y = negLogP, color = group)) +
  geom_point(size = 2.2, alpha = 0.85) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_hline(yintercept = -log10(p_threshold), linetype = "dashed") +
  scale_color_manual(values = c("HR > 1, P < 0.05" = "#B2182B",
                                "HR < 1, P < 0.05" = "#2166AC",
                                "Not significant" = "black")) +
  geom_text_repel(data = top_genes, aes(label = Gene), size = 4,
                  box.padding = 0.4, point.padding = 0.3, max.overlaps = 20) +
  labs(x = "Cox coefficient", y = "-log10(P value)", color = NULL) +
  theme_bw(base_size = 14)

ggsave("Cox_volcano.pdf", volcano, width = 7, height = 6)
