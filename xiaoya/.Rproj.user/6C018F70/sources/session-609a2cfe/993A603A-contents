library(data.table)
library(torch)
library(R6)
library(data.table)

EhrDataset <- R6::R6Class(
  classname = "EhrDataset",
  public = list(
    data = NULL,
    label = NULL,
    pid = NULL,

    initialize = function(data_path, mode = 'train') {
      self$data <- fread(file.path(data_path, paste0(mode, '_x.pkl')))
      self$label <- fread(file.path(data_path, paste0(mode, '_y.pkl')))
      self$pid <- fread(file.path(data_path, paste0(mode, '_pid.pkl')))
    },

    length = function() {
      nrow(self$label)
    },

    get_item = function(index) {
      list(data = self$data[index, ], label = self$label[index, ], pid = self$pid[index])
    }
  )
)


EhrDataModule <- R6::R6Class(
  classname = "EhrDataModule",
  public = list(
    train_dataset = NULL,
    val_dataset = NULL,
    test_dataset = NULL,
    initialize = function(data_path, batch_size = 32) {
      private$data_path <- data_path
      private$batch_size <- batch_size
    },
    setup = function(stage) {
      if (stage == "fit") {
        self$train_dataset <- EhrDataset$new(private$data_path, mode = "train")
        self$val_dataset <- EhrDataset$new(private$data_path, mode = "val")
      }
      if (stage == "test") {
        self$test_dataset <- EhrDataset$new(private$data_path, mode = "test")
      }
    },
    train_dataloader = function() {
      train_dataset <- self$train_dataset
      return(torch::DataLoader(train_dataset$data, train_dataset$label, batch_size = private$batch_size, shuffle = TRUE, collate_fn = self$pad_collate))
    },
    val_dataloader = function() {
      val_dataset <- self$val_dataset
      return(torch::DataLoader(val_dataset$data, val_dataset$label, batch_size = private$batch_size, shuffle = FALSE, collate_fn = self$pad_collate))
    },
    test_dataloader = function() {
      test_dataset <- self$test_dataset
      return(torch::DataLoader(test_dataset$data, test_dataset$label, batch_size = private$batch_size, shuffle = FALSE, collate_fn = self$pad_collate))
    },
    pad_collate = function(batch) {
      xx <- lapply(batch, function(x) unlist(x[[1]]))
      yy <- lapply(batch, function(x) unlist(x[[2]]))
      pid <- lapply(batch, function(x) x[[3]])
      lens <- as_tensor(sapply(xx, length))
      xx <- lapply(xx, as_tensor)
      yy <- lapply(yy, as_tensor)
      xx_pad <- torch::pad_sequence(xx, batch_first = TRUE, padding_value = 0)
      yy_pad <- torch::pad_sequence(yy, batch_first = TRUE, padding_value = 0)
      return(list(xx_pad, yy_pad, lens, pid))
    }
  ),
  private = list(
    data_path = NULL,
    batch_size = 32
  )
)


dataloader <- function(dataset, batch_size, shuffle = TRUE) {
  indices <- if (shuffle) sample(seq_len(dataset$length())) else seq_len(dataset$length())
  batches <- split(indices, ceiling(seq_along(indices) / batch_size))

  lapply(batches, function(batch_indices) {
    items <- lapply(batch_indices, dataset$get_item)
    list(data = do.call(rbind, lapply(items, function(x) x$data)),
         label = do.call(rbind, lapply(items, function(x) x$label)),
         pid = unlist(lapply(items, function(x) x$pid)))
  })
}

