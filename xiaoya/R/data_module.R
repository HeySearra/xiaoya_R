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
      x = self$data[[index]],  # 假设self$data是一个包含多个torch_tensor的列表
      y = self$target[[index]]
    )
  },

  .getitem = function(index) {
    list(
      x = self$data[[index]],  # 假设self$data是一个包含多个torch_tensor的列表
      y = self$target[[index]]
    )
  },

  .length = function() {
    length(self$data)  # 假设self$data是一个列表
  },

  length = function() {
    length(self$data)  # 假设self$data是一个列表
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


# test_data <- list(torch_tensor(matrix(1:9, nrow = 3)), torch_tensor(matrix(10:18, nrow = 3)))
# test_target <- list(torch_tensor(1), torch_tensor(2))
# test_dataset <- EhrDataset(test_data, test_target)
# print(test_dataset)
# print(test_dataset$get_item(1))

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
