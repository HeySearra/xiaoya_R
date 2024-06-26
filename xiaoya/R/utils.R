library(reticulate)


get_los_info <- function(dataset_dir) {
  path <- file.path(dataset_dir, 'los_info.pkl')
  los_info <- py_load_object(path)
  return(los_info)
}

# 定义函数 unpad_y
unpad_y <- function(y_pred, y_true, lens) {
  raw_device <- torch$device
  device <- torch$device("cpu")
  y_pred <- y_pred$to(device)
  y_true <- y_true$to(device)
  lens <- lens$to(device)
  y_pred_unpad <- unpad_sequence(y_pred, batch_first=TRUE, lengths=lens)
  y_pred_stack <- torch$vstack(y_pred_unpad)$squeeze(dim=-1)
  y_true_unpad <- unpad_sequence(y_true, batch_first=TRUE, lengths=lens)
  y_true_stack <- torch$vstack(y_true_unpad)$squeeze(dim=-1)
  return(list(y_pred_stack$to(raw_device), y_true_stack$to(raw_device)))
}

# 定义函数 unpad_batch
unpad_batch <- function(x, y, lens) {
  x <- x$detach()$cpu()
  y <- y$detach()$cpu()
  lens <- lens$detach()$cpu()
  x_unpad <- unpad_sequence(x, batch_first=TRUE, lengths=lens)
  x_stack <- torch$vstack(x_unpad)$squeeze(dim=-1)
  y_unpad <- unpad_sequence(y, batch_first=TRUE, lengths=lens)
  y_stack <- torch$vstack(y_unpad)$squeeze(dim=-1)
  return(list(x_stack$numpy(), y_stack$numpy()))
}
