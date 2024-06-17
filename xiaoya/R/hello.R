# Hello, world!
#
# This is an example function named 'hello'
# which prints 'Hello, world!'.
#
# You can learn more about package authoring with RStudio at:
#
#   http://r-pkgs.had.co.nz/
#
# Some useful keyboard shortcuts for package authoring:
#
#   Install Package:           'Ctrl + Shift + B'
#   Check Package:             'Ctrl + Shift + E'
#   Test Package:              'Ctrl + Shift + T'
library(data.table)
source('R/data_handler.R')
source('R/pipeline.R')

labtest_data <- fread('datasets/raw_labtest_data.csv')
events_data <- fread('datasets/raw_events_data.csv')
target_data <- fread('datasets/raw_target_data.csv')
data_path <- 'datasets'

# 初始化DataHandler对象
data_handler <- init_data_handler(labtest_data, events_data, target_data, data_path)
debug(format_and_merge_dataframes)
debug(format_dataframe)
# 执行数据处理操作
data_handler <- format_and_merge_dataframes(data_handler)
debug(save_processed_data)
debug(merge_dataframes)
data_handler <- merge_dataframes(data_handler)

data_handler <- save_processed_data(data_handler)
# standard_labtest_data.csv
# standard_events_data.csv
# standard_target_data.csv
# standard_merged_data.csv

debug(analyze_dataset)
result <- analyze_dataset(data_handler)

# 初始化Pipeline对象并执行
pipeline <- Pipeline$new()
debug(pipeline$execute)
result <- pipeline$execute()
print(result)

# data analysis
data_analyzer <- DataAnalyzer$new(config, model_path)
result1 <- data_analyzer$adaptive_feature_importance(df, x, patient_index = 1)
