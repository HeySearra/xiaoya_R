library(torch)
library(R6)

GRU <- nn_module(
  "GRU",

  initialize = function(input_dim, hidden_dim, act_layer, drop, ...) {
    self$input_dim <- input_dim
    self$hidden_dim <- hidden_dim
    self$proj <- nn_linear(input_dim, hidden_dim)
    self$act <- act_layer()
    self$dropout <- nn_dropout(p = drop)
    self$gru <- nn_gru(input_size = hidden_dim, hidden_size = hidden_dim, num_layers = 1, batch_first = TRUE)
    self$output <- nn_linear(hidden_dim, 1)
  },

  forward = function(x) {
    x <- self$proj(x)
    x <- self$act(x)
    x <- self$dropout(x)
    x <- self$gru(x)
    x <- self$output(x[[1]][, -1, , drop = FALSE])
    return(x)
  }
)

MLP <- nn_module(
  "MLP",

  initialize = function(input_dim, hidden_dim, act_layer, drop, ...) {
    self$input_dim <- input_dim
    self$hidden_dim <- hidden_dim
    self$proj <- nn_linear(input_dim, hidden_dim)
    self$act <- act_layer()
    self$dropout <- nn_dropout(p = drop)
    self$output <- nn_linear(hidden_dim, 1)
  },

  forward = function(x) {
    x <- self$proj(x)
    x <- self$act(x)
    x <- self$dropout(x)
    x <- self$output(x)
    return(x)
  }
)
