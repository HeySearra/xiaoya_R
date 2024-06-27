library(torch)
library(R6)

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
my_gru_instance <- MyGRU$new(input_dim = 10, hidden_dim = 20, act_layer = "relu", drop = 0.5)
print(my_gru_instance)

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
