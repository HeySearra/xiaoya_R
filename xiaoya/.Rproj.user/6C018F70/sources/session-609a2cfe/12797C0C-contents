library(torch)
library(R6)

GRU <- R6Class(
  "GRU",
  inherit = torch$nn$Module,

  public = list(
    initialize = function(input_dim, hidden_dim, act_layer, drop, ...) {
      super$initialize()
      self$input_dim <- input_dim
      self$hidden_dim <- hidden_dim

      self$proj <- torch$nn$Linear(input_dim, hidden_dim)
      self$act <- act_layer$new()
      self$gru <- torch$nn$GRU(input_size = input_dim, hidden_size = hidden_dim, num_layers = 1, batch_first = TRUE)
    },

    forward = function(x, ...) {
      x <- self$proj(x)
      x <- self$act(x)
      x <- torch$nn$functional$dropout(x, ...)
      x <- self$gru(x)
      return(x)
    }
  )
)
