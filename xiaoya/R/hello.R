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


data_handler <- list(
  merged_df = fread("datasets/standard_merged_data.csv"),
  target = fread("datasets/standard_target_data.csv")
)

# 初始化Pipeline对象并执行
debug(Pipeline$new)
pipeline_gru <- Pipeline$new(data_handler, model_type = "GRU", act_function = "relu")
debug(pipeline_gru$execute)
result_gru <- pipeline_gru$execute()
print(result_gru)

# data analysis
data_analyzer <- DataAnalyzer$new(config, model_path)
result1 <- data_analyzer$adaptive_feature_importance(df, x, patient_index = 1)



# 创建示例数据
data <- torch_tensor(matrix(rnorm(1000), ncol = 10), dtype = torch_float())
target <- torch_tensor(rnorm(100), dtype = torch_float())

# 初始化数据处理模块
data_handler <- list(
  merged_df = data.table(matrix(rnorm(1000), ncol = 10)),
  target = data.table(Outcome = rnorm(100))
)

# 初始化 Pipeline 实例
debug(Pipeline$new)
pipeline_gru <- Pipeline$new(data_handler, model_type = "GRU", act_function = "relu")
debug(pipeline_gru$train)
# 训练模型
pipeline_gru$train()

# 示例使用
data_handler <- list(
  train_dataloader = function() {
    # 返回一个 dataloader 示例
  }
)

# 创建 GRU 模型
pipeline_gru <- Pipeline$new(data_handler, model_type = "GRU", act_function = "relu")
pipeline_gru$train()

# 创建 MLP 模型
pipeline_mlp <- Pipeline$new(data_handler, model_type = "MLP", act_function = "relu")
pipeline_mlp$train()






###################################

library(data.table)
library(torch)

# 读取数据
merged_df <- fread("datasets/standard_merged_data.csv")
target <- fread("datasets/standard_target_data.csv")

# 将 target 数据添加到 merged_df
merged_df <- cbind(merged_df, target)
EHRDataset <- dataset(
  name = "EHRDataset",

  initialize = function(data) {
    self$data <- data

    # 假设 data 的最后一列是 target
    self$x <- data[, -ncol(data), with = FALSE]
    self$y <- data[, ncol(data), with = FALSE]

    self$x <- as.matrix(self$x)
    self$y <- as.matrix(self$y)
  },

  .getitem = function(i) {
    list(x = torch_tensor(self$x[i, ]), y = torch_tensor(self$y[i]))
  },

  .length = function() {
    nrow(self$data)
  }
)

data_handler <- list(
  train_dataloader = function(batch_size = 16, shuffle = TRUE) {
    # 读取和预处理数据
    merged_df <- fread("datasets/standard_merged_data.csv")
    target <- fread("datasets/standard_target_data.csv")
    merged_df <- cbind(merged_df, target)

    # 分割训练和验证数据集
    set.seed(123)
    train_indices <- sample(1:nrow(merged_df), 0.8 * nrow(merged_df))
    train_data <- merged_df[train_indices, ]
    valid_data <- merged_df[-train_indices, ]

    # 创建数据集
    train_ds <- EHRDataset(train_data)
    valid_ds <- EHRDataset(valid_data)

    # 创建数据加载器
    train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = shuffle)
    valid_dl <- dataloader(valid_ds, batch_size = batch_size, shuffle = FALSE)

    list(train_dl = train_dl, valid_dl = valid_dl)
  }
)

# 示例使用
dataloaders <- data_handler$train_dataloader()
train_dl <- dataloaders$train_dl
valid_dl <- dataloaders$valid_dl


str(dataloaders)

