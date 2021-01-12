# https://blogs.rstudio.com/ai/posts/2020-11-03-torch-tabular/

setwd('/home/ryan/Studies/2020ResearchTraining/ranknet/scripts/rtorch/')

# Load Packages -----------------------------------------------------------
library(torch)
library(purrr)
library(readr)
library(dplyr)
library(ggplot2)
library(ggrepel)


# Download Data -----------------------------------------------------------

if (!file.exists('agaricus-lepiota.data')) {
  download.file(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
    destfile = "agaricus-lepiota.data"
  )
} 



# Rename the data