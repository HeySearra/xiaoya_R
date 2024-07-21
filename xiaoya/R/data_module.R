library(data.table)
library(torch)
library(R6)
library(coro)
library(data.table)
library(reticulate)
use_python("C:/Users/lenovo/AppData/Local/Programs/Python/Python38/python.exe")

EhrDataset <- dataset(
  name = "EhrDataset",

  initialize = function(data, target) {
    self$data <- data
    self$target <- target
  },

  get_item = function(index) {
    list(
      x = torch_tensor(as.numeric(self$data[index, ]), dtype = torch_float()),
      y = torch_tensor(as.numeric(self$target[index]), dtype = torch_float())
    )
  },
  .getitem = function(index) {
    list(
      x = torch_tensor(as.numeric(self$data[index, ]), dtype = torch_float()),
      y = torch_tensor(as.numeric(self$target[index]), dtype = torch_float())
    )
  },

  .length = function() {
    nrow(self$data)
  },

  length = function() {
    nrow(self$data)
  }
)


EhrDataModule <- R6Class(
  "EhrDataModule",

  public = list(
    data = NULL,
    target = NULL,
    batch_size = 16,

    initialize = function(data, target, batch_size) {
      self$data <- data
      self$target <- target
      self$batch_size <- batch_size
    },

    train_dataloader = function() {
      dataset_instance <- EhrDataset(self$data, self$target)
      dataloader <- dataloader(dataset_instance, batch_size = self$batch_size, shuffle = TRUE)
      dataloader
    }
  )
)


# data <- torch_tensor(matrix(rnorm(1000), ncol = 10), dtype = torch_float())
# target <- torch_tensor(rnorm(100), dtype = torch_float())

# 初始化 EhrDataModule 实例
# data_module <- EhrDataModule$new(data = data, target = target, batch_size = 4)
# debug(data_module$train_dataloader)
# 测试 train_dataloader 方法
# dataloader <- data_module$train_dataloader()

# 打印数据加载器中的第一个批次
# first_batch <- dataloader$.iter()$.next()
# print(first_batch)

##############################

# 示例调用
# merged_df <- fread("datasets/standard_merged_data.csv")
# target <- fread("datasets/standard_target_data.csv")

# 将 target 数据添加到 merged_df
# merged_df <- cbind(merged_df, target)

# 假设最后一列是 target
# data <- merged_df[, -ncol(merged_df), with = FALSE]
# target <- merged_df[, ncol(merged_df), with = FALSE]

# data_matrix <- as.matrix(data)
# target_vector <- as.matrix(target)

# data_module <- EhrDataModule$new(data = data_matrix, target = target_vector, batch_size = 16)
# train_dl <- data_module$train_dataloader()

# 打印数据加载器中的一个批次
# coro::loop(for (batch in train_dl) {
#   print(batch)
#   break
# })
