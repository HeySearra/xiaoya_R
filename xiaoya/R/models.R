library(torch)
library(R6)

GRU <- torch::nn_module(
  "GRU",
  initialize = function(input_dim, hidden_dim, act_layer, drop) {
    self$gru <- nn_gru(input_dim, hidden_dim, batch_first = TRUE)
    self$fc <- nn_linear(hidden_dim, 1)
    self$activation <- act_layer()
    self$dropout <- nn_dropout(drop)
  },
  forward = function(x) {
    gru_output <- self$gru(x)
    x <- gru_output[[1]]
    x <- x[, -1, ]
    x <- self$dropout(x)
    x <- self$fc(x)
    x <- self$activation(x)
    x
  }
)


# 示例测试
# input_dim <- 75
# hidden_dim <- 32
# dropout_rate <- 0.5
# model <- GRU$new(input_dim = input_dim, hidden_dim = hidden_dim, act_layer = nn_relu, drop = dropout_rate)

# 创建一个模拟输入
# input_tensor <- torch_randn(c(32, 75, input_dim)) # (batch_size, seq_len, input_dim)

# 前向传播
# output <- model$forward(input_tensor)
# print(paste("Output shape:", paste(dim(output), collapse = "x")))


MyGRU <- R6::R6Class(
  "MyGRU",
  public = list(
    input_dim = NULL,
    hidden_dim = NULL,
    act_layer = NULL,
    drop = NULL,

    initialize = function(input_dim, hidden_dim, act_layer, drop) {
      self$input_dim <- input_dim
      self$hidden_dim <- hidden_dim
      self$act_layer <- act_layer
      self$drop <- drop
    },

    forward = function(x) {
      # Forward pass logic
    }
  )
)

# 测试 MyGRU 类的实例化
# my_gru_instance <- MyGRU$new(input_dim = 10, hidden_dim = 20, act_layer = "relu", drop = 0.5)
# print(my_gru_instance)

# 定义 MyMLP 类
MyMLP <- R6::R6Class(
  "MyMLP",
  public = list(
    input_dim = NULL,
    hidden_dim = NULL,
    act_layer = NULL,
    drop = NULL,

    initialize = function(input_dim, hidden_dim, act_layer, drop) {
      self$input_dim <- input_dim
      self$hidden_dim <- hidden_dim
      self$act_layer <- act_layer
      self$drop <- drop
    },

    forward = function(x) {
      # Forward pass logic for MLP
    }
  )
)
