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

  .getitem = function(index) {
    list(x = self$data[index, ], y = self$target[index])
  },

  .length = function() {
    self$data$size(1)
  }
)


EhrDataModule <- R6Class(
  "EhrDataModule",

  public = list(
    data = NULL,
    target = NULL,
    batch_size = NULL,

    initialize = function(data, target, batch_size) {
      self$data <- data
      self$target <- target
      self$batch_size <- batch_size
    },

    train_dataloader = function() {
      dataset_instance <- EhrDataset(self$data, self$target)
      dataloader <- dataloader(dataset_instance, batch_size = self$batch_size, shuffle = TRUE)
      return(dataloader)
    }
  )
)



# 创建示例数据
data <- torch_tensor(matrix(rnorm(1000), ncol = 10), dtype = torch_float())
target <- torch_tensor(rnorm(100), dtype = torch_float())

# 初始化 EhrDataModule 实例
data_module <- EhrDataModule$new(data = data, target = target, batch_size = 4)
debug(data_module$train_dataloader)
# 测试 train_dataloader 方法
dataloader <- data_module$train_dataloader()

# 打印数据加载器中的第一个批次
first_batch <- dataloader$.iter()$.next()
print(first_batch)
