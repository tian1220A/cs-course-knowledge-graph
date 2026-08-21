# ===================== 【完全可运行 · 无错版】 =====================
install.packages(c("tidyverse","ggplot2","RColorBrewer","vegan","devtools"))
if (!requireNamespace("linkET", quietly = TRUE)) {
  devtools::install_github("Hy4m/linkET", upgrade = "never")
}

library(linkET)
library(tidyverse)
library(ggplot2)
library(RColorBrewer)
library(vegan)


data("varechem", package = "vegan")
data("varespec", package = "vegan")


mdata <- correlate(varechem)

mantel <- mantel_test(
  varespec, varechem,
  spec_select = list(
    Spec01 = 1:7,
    Spec02 = 8:18,
    Spec03 = 19:37,
    Spec04 = 38:44
  )
)

mantel <- mantel %>%
  mutate(
    rd = cut(r, breaks = c(-Inf, 0.2, 0.4, Inf), labels = c("< 0.2", "0.2 - 0.4", ">= 0.4")),
    pd = cut(p, breaks = c(-Inf, 0.01, 0.05, Inf), labels = c("< 0.01", "0.01 - 0.05", ">= 0.05"))
  )


p <- qcorrplot(mdata, type = "lower", diag = FALSE) +
  geom_square() +
  geom_couple(aes(colour = pd, size = rd), data = mantel, curvature = nice_curvature()) +
  scale_fill_gradientn(colours = brewer.pal(11, "RdYlGn")) +
  scale_size_manual(values = c(0.5, 1, 2)) +
  scale_colour_manual(values = c('#7AA15E','#B7B663','#CFCECC')) +
  guides(
    size = guide_legend(title = "Mantel's r", override.aes = list(colour = "grey35"), order = 2),
    colour = guide_legend(title = "Mantel's p", override.aes = list(size = 3), order = 1),
    fill = guide_colorbar(title = "Pearson's r", order = 3)
  ) +
  labs(title = "网络热图") +
  theme(plot.title = element_text(hjust = 0.5))

print(p)
ggsave("network_plot.png", p, width = 10, height = 8, dpi = 300)