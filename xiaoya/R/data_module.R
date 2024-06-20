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
      dataset <- tensor_dataset(tensors = list(self$data, self$target))
      dataloader(dataset, batch_size = self$batch_size, shuffle = TRUE)
    }
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

