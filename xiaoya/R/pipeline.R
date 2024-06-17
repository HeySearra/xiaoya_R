library(data.table)
library(torch)
source('R/utils.R')
source("R/data_module.R")

# 使用：
# labtest_data <- fread('datasets/raw_labtest_data.csv')
# events_data <- fread('datasets/raw_events_data.csv')
# target_data <- fread('datasets/raw_target_data.csv')

# 初始化DataHandler对象
# data_handler <- init_data_handler(labtest_data, events_data, target_data, data_path)

# 执行数据处理操作
# data_handler <- format_and_merge_dataframes(data_handler)
# data_handler <- save_processed_data(data_handler)

# 初始化Pipeline对象并执行
# pipeline <- Pipeline$new()
# result <- pipeline$execute()
# print(result)

Pipeline <- R6::R6Class(
  "Pipeline",

  public = list(
    config = list(),
    data_path = NULL,
    los_info = NULL,
    model_path = NULL,

    initialize = function(model = 'GRU',
                          batch_size = 64,
                          learning_rate = 0.001,
                          hidden_dim = 32,
                          epochs = 50,
                          patience = 10,
                          task = 'multitask',
                          seed = 42,
                          data_path = "./datasets",
                          demographic_dim = 2,
                          labtest_dim = 73) {

      self$config <- list(
        model = model,
        batch_size = batch_size,
        learning_rate = learning_rate,
        hidden_dim = hidden_dim,
        output_dim = 1,
        epochs = epochs,
        patience = patience,
        task = task,
        seed = seed,
        demo_dim = demographic_dim,
        lab_dim = labtest_dim
      )

      self$data_path <- data_path
      self$los_info <- get_los_info(data_path)
    },

    train = function(ckpt_path = './checkpoints', ckpt_name = 'best') {
      main_metric <- ifelse(self$config$task %in% c('outcome', 'multitask'), 'auprc', 'mae')
      mode <- ifelse(self$config$task %in% c('outcome', 'multitask'), 'max', 'min')

      self$config <- c(self$config, list(los_info = self$los_info, main_metric = main_metric, mode = mode))

      dm <- EhrDataModule$new(data_path = self$data_path, batch_size = self$config$batch_size)

      ckpt_url <- file.path(ckpt_path, self$config$task, paste0(self$config$model, '-seed', self$config$seed))

      early_stopping_callback <- EarlyStopping(monitor = main_metric, patience = self$config$patience, mode = mode)
      checkpoint_callback <- ModelCheckpoint(monitor = main_metric, mode = mode, dirpath = ckpt_url, filename = ckpt_name)

      set.seed(self$config$seed)
      torch.manual_seed(self$config$seed)

      accelerator <- ifelse(torch_cuda_is_available(), 'gpu', 'cpu')
      devices <- ifelse(accelerator == 'gpu', list(0), list(1))

      pipeline <- DlPipeline$new(self$config)
      trainer <- Trainer$new(accelerator = accelerator, gpus = devices, max_epochs = self$config$epochs, callbacks = list(early_stopping_callback, checkpoint_callback), logger = FALSE, progress_bar_refresh_rate = 1)
      trainer$fit(pipeline, datamodule = dm)
      self$model_path <- checkpoint_callback$best_model_path
    },

    predict = function(self, model_path, metric_path = './metrics') {
      self$config <- c(self$config, list(los_info = self$los_info))

      # data
      dm <- EhrDataModule(data_path = self$data_path, batch_size = self$config$batch_size)

      # device
      accelerator <- ifelse(torch_cuda_is_available(), 'gpu', 'cpu')
      devices <- ifelse(accelerator == 'gpu', list(0), list(1))

      # train/val/test
      pipeline <- DlPipeline$new(self$config)
      trainer <- Trainer$new(accelerator = accelerator, gpus = devices, max_epochs = 1, logger = FALSE, num_sanity_val_steps = 0)
      trainer$test(pipeline, datamodule = dm, ckpt_path = model_path)

      performance <- as.list(pipeline$test_performance)
      performance <- lapply(performance, function(x) as.numeric(x))
      ckpt_name <- gsub('ckpt', 'csv', basename(model_path))
      metric_url <- file.path(metric_path, self$config$task, paste0(self$config$model, '-seed', self$config$seed))
      dir.create(metric_url, recursive = TRUE, showWarnings = FALSE)
      write.csv(data.frame(performance), file.path(metric_url, ckpt_name), row.names = FALSE)

      output <- as.list(pipeline$test_outputs)
      return(list(detail = list(
        preds = output$preds,
        labels = output$labels,
        config = self$config,
        performance = performance
      )))
    },

    execute = function(model_path = NULL) {
      if (is.null(model_path)) {
        self$train()
        model_path <- self$model_path
      }
      return(self$predict(model_path = self$model_path))
    }

  )
)
