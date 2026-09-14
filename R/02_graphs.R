# Exploratory plots for the seven-GGUF DS-1000 comparison.
# Approach: ggplot2 + theme_wong, as in
# https://github.com/semenoffalex/graph-memory-starter/blob/main/R/02_graphs.R

library(ggplot2)
library(dplyr)
library(readr)
library(forcats)

# TODO: make a categorical color scale for a fixed number of cathegories (see Wong for the recomendation of the maximum number)

source("https://raw.githubusercontent.com/semenoffalex/MiscR/main/theme_wong.R")
wong_set_style("sf")

repo <- if (dir.exists("results")) "." else ".."
out_dir <- file.path(repo, "R", "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

model_levels <- c(
  "REAP20-28B",
  "35B-A3B",
  "GSQ-RCO",
  "3.8 UD-Q4_K_M",
  "27B UD-Q4_K_XL",
  "Ridge IQ2_M",
  "Coder-30B"
)

recoded <- c(
  "barozp-Qwen3.6-28B-REAP20-A3B" = "REAP20-28B",
  "unsloth-Qwen3.6-35B-A3B" = "35B-A3B",
  "ISTA-DASLab-Qwen3.8-27B-GSQ-RCO-GGUF" = "GSQ-RCO",
  "unsloth-Qwen3.8-27B-UD-Q4_K_M" = "3.8 UD-Q4_K_M",
  "unsloth-Qwen3.6-27B-UD-Q4_K_XL" = "27B UD-Q4_K_XL",
  "empero-ai-Qwen3.8-27B-Ridge" = "Ridge IQ2_M",
  "unsloth-Qwen3-Coder-30B-A3B" = "Coder-30B"
)

d <- read_csv(file.path(repo, "results", "comparison-summary.csv"), show_col_types = FALSE) |>
  filter(model %in% names(recoded)) |>
  mutate(
    model = recode(model, !!!as.list(recoded)),
    model = factor(model, levels = model_levels),
    pass_pct = round(pass_rate * 100, 1),
    gen_tok_s = round(gen_tok_s_mean, 1),
    completion_tokens = round(completion_tokens_mean)
  )

lib_n <- c(
  Matplotlib = 155, Numpy = 220, Pandas = 291, Pytorch = 68,
  Scipy = 106, Sklearn = 115, Tensorflow = 45
)

by_lib <- read_csv(file.path(repo, "results", "comparison-by-library.csv"), show_col_types = FALSE) |>
  filter(model %in% names(recoded)) |>
  mutate(
    model = recode(model, !!!as.list(recoded)),
    model = factor(model, levels = model_levels),
    library = factor(library, levels = names(lib_n)),
    pass_pct = round(pass_rate * 100, 1)
  )

theme_rank <- function() {
  theme_wong(style = "sf", scales = FALSE, grid = FALSE) +
    theme(
      axis.text.y = element_text(colour = wong_colors$ink),
      axis.line.x = element_blank(),
      axis.ticks.x = element_blank(),
      panel.grid.major.x = element_line(colour = wong_colors$grid, linewidth = 0.3),
      panel.grid.major.y = element_blank()
    )
}

save_plot <- function(p, name, width = 8, height = 5.2) {
  ggsave(file.path(out_dir, paste0(name, ".png")), p, width = width, height = height, dpi = 150)
  p
}

## Graph: DS-1000 pass rate
p_pass <- ggplot(d, aes(pass_pct, fct_rev(model))) +
  geom_col(width = 0.72) +
  geom_vline(xintercept = 38.8, colour = wong_colors$axis, linewidth = 0.4) +
  geom_text(
    aes(label = sprintf("%.1f", pass_pct)),
    hjust = -0.2,
    size = 3.2
  ) +
  scale_x_continuous(
    limits = c(0, 78),
    expand = expansion(mult = c(0, 0.04))
  ) +
  labs(
    title = "DS-1000 pass rate",
    subtitle = "Share of 1000 problems whose generated code passed. Vertical line: Codex-002 38.8%.",
    x = NULL,
    y = NULL
  ) +
  theme_rank()

## Graph: Generation speed
p_speed <- ggplot(d, aes(gen_tok_s, fct_rev(model))) +
  geom_col(width = 0.72) +
  geom_text(
    aes(label = sprintf("%.1f", gen_tok_s)),
    hjust = -0.2,
    size = 3.2
  ) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title = "Generation speed (tok/s)",
    subtitle = "Mean llama.cpp predicted_per_second across 1000 requests.",
    x = NULL,
    y = NULL
  ) +
  theme_rank()

## Graph: Completion tokens
p_tok <- ggplot(d, aes(completion_tokens, fct_rev(model))) +
  geom_col(width = 0.72) +
  geom_text(
    aes(label = completion_tokens),
    hjust = -0.2,
    size = 3.2
  ) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title = "Completion tokens (mean)",
    subtitle = "Mean generated tokens per problem. Prompt is 369 except Coder (352).",
    x = NULL,
    y = NULL
  ) +
  theme_rank()

## Graph: Pass rate by library
p_lib <- ggplot(by_lib, aes(pass_pct, fct_rev(model))) +
  geom_col(width = 0.72) +
  geom_text(
    aes(label = sprintf("%.1f", pass_pct)),
    hjust = -0.15,
    size = 2.8
  ) +
  facet_wrap(~library, ncol = 4) +
  scale_x_continuous(
    limits = c(0, 100),
    expand = expansion(mult = c(0, 0.08))
  ) +
  labs(
    title = "Pass rate by library (%)",
    subtitle = "Per-library fraction of passing problems. n: Matplotlib 155, Numpy 220, Pandas 291, Pytorch 68, Scipy 106, Sklearn 115, Tensorflow 45.",
    x = NULL,
    y = NULL
  ) +
  theme_rank()

eval_paths <- list.files(file.path(repo, "results"), pattern = "-eval\\.csv$", full.names = TRUE)
by_pert <- bind_rows(lapply(eval_paths, function(path) {
  tag <- sub("-eval\\.csv$", "", basename(path))
  if (!tag %in% names(recoded)) {
    return(NULL)
  }
  read_csv(path, show_col_types = FALSE) |>
    mutate(model = recoded[[tag]])
})) |>
  group_by(model, perturbation_type) |>
  summarise(pass_pct = round(mean(score) * 100, 1), n = n(), .groups = "drop") |>
  mutate(
    model = factor(model, levels = model_levels),
    perturbation_type = factor(
      perturbation_type,
      levels = c("Origin", "Semantic", "Surface", "Difficult-Rewrite")
    )
  )

## Graph: Pass rate by perturbation
p_pert <- ggplot(by_pert, aes(pass_pct, fct_rev(model))) +
  geom_col(width = 0.72) +
  geom_text(
    aes(label = sprintf("%.1f", pass_pct)),
    hjust = -0.15,
    size = 2.8
  ) +
  facet_wrap(~perturbation_type, ncol = 2) +
  scale_x_continuous(
    limits = c(0, 80),
    expand = expansion(mult = c(0, 0.08))
  ) +
  labs(
    title = "Pass rate by perturbation (%)",
    subtitle = "n: Origin 452, Semantic 234, Surface 152, Difficult-Rewrite 162.",
    x = NULL,
    y = NULL
  ) +
  theme_rank()

save_plot(p_pass, "ds1000-pass-rate")
save_plot(p_speed, "ds1000-gen-tok-s")
save_plot(p_tok, "ds1000-completion-tokens")
save_plot(p_lib, "ds1000-pass-by-library", width = 10, height = 7.2)
save_plot(p_pert, "ds1000-pass-by-perturbation", width = 9, height = 6.4)

if (interactive()) {
  print(p_pass)
  print(p_speed)
  print(p_tok)
  print(p_lib)
  print(p_pert)
}
