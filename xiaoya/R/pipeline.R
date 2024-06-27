library(data.table)
library(torch)
library(coro)
library(R6)
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
    data_handler = NULL,
    model = NULL,
    config = NULL,

    initialize = function(data_handler, model_type, act_function) {
      self$data_handler <- data_handler

      self$config <- list(
        model_type = model_type,
        act_layer = act_function,
        hidden_dim = 20,  # Example value
        drop = 0.5,       # Example value
        lr = 0.001        # Example value
      )

      input_dim <- ncol(merged_df) - 1  # Adjust to match your dataset

      if (model_type == "GRU") {
        self$model <- MyGRU$new(input_dim = input_dim, hidden_dim = self$config$hidden_dim,
                                act_layer = self$config$act_layer, drop = self$config$drop)
      } else if (model_type == "MLP") {
        self$model <- MyMLP$new(input_dim = input_dim, hidden_dim = self$config$hidden_dim,
                                act_layer = self$config$act_layer, drop = self$config$drop)
      } else {
        stop("Unsupported model type")
      }
    },

    train = function() {
      # 获取 dataloader
      dataloaders <- self$data_handler$train_dataloader()
      train_dl <- dataloaders$train_dl
      valid_dl <- dataloaders$valid_dl

      # 检查 model 参数是否为 NULL
      if (is.null(self$model$parameters)) {
        stop("Model parameters are NULL.")
      }

      # 定义优化器
      optimizer <- optim_adam(self$model$parameters, lr = self$config$lr)

      # 训练逻辑
      for (epoch in 1:20) {
        self$model$train()
        train_losses <- c()

        coro::loop(for (b in train_dl) {
          optimizer$zero_grad()
          output <- self$model$forward(b$x)
          loss <- nnf_mse_loss(output, b$y)

          loss$backward()
          optimizer$step()

          train_losses <- c(train_losses, loss$item())
        })

        cat(sprintf("Loss at epoch %d: %3.3f\n", epoch, mean(train_losses)))
      }
    }
  )
)

