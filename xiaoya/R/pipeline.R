library(data.table)
library(torch)
library(coro)
library(R6)
library(h2o)
library(reticulate)
library(dplyr)
source("R/models.R")
use_python("C:/Users/lenovo/AppData/Local/Programs/Python/Python38/python.exe")

activation_functions <- list(
  relu = nnf_relu,
  sigmoid = nn_sigmoid,
  tanh = nn_tanh
)


Pipeline <- R6::R6Class(
  "Pipeline",
  public = list(
    train_h2o = NULL,
    valid_h2o = NULL,
    model = NULL,
    config = NULL,

    initialize = function(data_handler, model_type, act_function) {
      Sys.setenv(JAVA_HOME = "D:/Program Files/Java/jdk-11.0.12")
      h2o.init()
      self$train_h2o <- data_handler$train_h2o
      self$valid_h2o <- data_handler$valid_h2o

      self$config <- list(
        model_type = model_type,
        act_layer = act_function,
        hidden_dim = 50,  # 示例值
        drop = 0.5,       # 示例值
        lr = 0.001        # 示例值
      )


      if (model_type == "GRU") {
        self$model <- h2o.deeplearning(
          x <- colnames(self$train_h2o)[!(colnames(self$train_h2o) %in% c("Outcome", "LOS"))],  # 特征列
          y = "Outcome",  # 目标列
          training_frame = self$train_h2o,
          validation_frame = self$valid_h2o,
          activation = "RectifierWithDropout",
          hidden = c(self$config$hidden_dim, self$config$hidden_dim, self$config$hidden_dim),  # 隐藏层大小
          epochs = 20,
          variable_importances = TRUE
        )
      } else if (model_type == "MLP") {
        self$model <- h2o.deeplearning(
          x <- colnames(self$train_h2o)[!(colnames(self$train_h2o) %in% c("Outcome", "LOS"))],  # 特征列
          y = "Outcome",  # 目标列
          training_frame = self$train_h2o,
          validation_frame = self$valid_h2o,
          activation = "Rectifier",
          hidden = c(self$config$hidden_dim, self$config$hidden_dim),  # 隐藏层大小
          epochs = 20,
          variable_importances = TRUE
        )
      } else {
        stop("Unsupported model type")
      }
    },

    train = function() {
      # 打印模型性能
      print(h2o.performance(self$model, valid = TRUE))
    }
  )
)
