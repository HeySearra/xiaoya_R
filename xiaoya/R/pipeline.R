library(data.table)
library(torch)
source('R/utils.R')
source("R/data_module.R")
source('R/models.R')

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

# 定义激活函数字典
activation_functions <- list(
  relu = nn_relu,
  sigmoid = nn_sigmoid,
  tanh = nn_tanh
)


Pipeline <- R6Class(
  "Pipeline",

  public = list(
    data_handler = NULL,
    model = NULL,
    data_module = NULL,
    config = NULL,
    model_path = NULL,

    initialize = function(data_handler, model_type = "GRU", act_function = "relu") {
      self$data_handler <- data_handler
      self$config <- list(
        batch_size = 32,
        epochs = 100,
        lr = 0.001,
        hidden_dim = 64,
        act_layer = activation_functions[[act_function]],
        drop = 0.2
      )

      # 数据预处理
      self$prepare_data()
      self$build_model(model_type)
    },

    prepare_data = function() {
      # 获取特征数据
      features <- self$data_handler$merged_df[, -c("PatientID", "RecordTime", "Outcome", "LOS"), with = FALSE]
      target <- self$data_handler$target[, .(PatientID, Outcome)] # 只保留 PatientID 和 Outcome

      # 确保特征数据和目标数据的患者ID一致
      patient_ids <- intersect(unique(self$data_handler$merged_df$PatientID), unique(self$data_handler$target$PatientID))

      # 过滤特征数据和目标数据
      filtered_data <- self$data_handler$merged_df[self$data_handler$merged_df$PatientID %in% patient_ids]
      filtered_target <- self$data_handler$target[self$data_handler$target$PatientID %in% patient_ids]

      # 将特征数据和目标数据转换为矩阵
      features_filtered <- filtered_data[, -c("PatientID", "RecordTime", "Outcome", "LOS"), with = FALSE]
      target_filtered <- filtered_target[, .(PatientID, Outcome)]

      numeric_matrix <- as.matrix(sapply(features_filtered, as.numeric))
      target_tensor <- torch_tensor(as.numeric(target_filtered$Outcome), dtype = torch_float())

      # 按患者ID分组
      patient_data <- split(filtered_data, filtered_data$PatientID)
      patient_target <- split(filtered_target, filtered_target$PatientID) # 分组目标数据

      # 将每个患者的数据转换为矩阵，并处理缺失值
      patient_matrices <- lapply(patient_data, function(df) {
        df <- df[, -c("PatientID", "RecordTime", "Outcome", "LOS"), with = FALSE]
        df <- as.data.frame(lapply(df, function(col) {
          col[is.na(col)] <- 0  # 替换缺失值为0
          as.numeric(col)
        }))
        as.matrix(df)
      })

      # 处理目标数据
      patient_targets <- lapply(patient_target, function(df) {
        outcome <- df$Outcome
        outcome <- as.numeric(outcome)
        return(outcome)
      })

      # 转换为tensor
      numeric_tensors <- lapply(patient_matrices, function(mat) {
        torch_tensor(mat, dtype = torch_float())
      })

      target_tensors <- lapply(patient_targets, function(target) {
        torch_tensor(target, dtype = torch_float())
      })

      # 构建 EhrDataModule 实例
      self$data_module <- EhrDataModule$new(data = numeric_tensors, target = target_tensors, batch_size = self$config$batch_size)
    },

    build_model = function(model_type) {
      input_dim <- ncol(self$data_handler$merged_df) - 4  # 减去PatientID, RecordTime, Outcome, LOS的列数

      if (model_type == "GRU") {
        self$model <- GRU$new(
          input_dim = input_dim,
          hidden_dim = self$config$hidden_dim,
          act_layer = self$config$act_layer,
          drop = self$config$drop
        )
      } else if (model_type == "MLP") {
        self$model <- MLP$new(
          input_dim = input_dim,
          hidden_dim = self$config$hidden_dim,
          act_layer = self$config$act_layer,
          drop = self$config$drop
        )
      } else {
        stop("Unsupported model type.")
      }
    },

    train = function() {
      optimizer <- optim_adam(self$model$parameters, lr = self$config$lr)
      criterion <- nn_mse_loss()

      for (epoch in seq_len(self$config$epochs)) {
        self$model$train()
        dataloader <- self$data_module$train_dataloader()

        total_loss <- 0

        coro::loop(for (batch in dataloader) {
          x <- batch[[1]]
          y <- batch[[2]]

          print(paste("Input shape:", paste(dim(x), collapse = "x")))
          print(paste("Target shape:", paste(dim(y), collapse = "x")))

          output <- self$model$forward(x)

          print(paste("Output shape:", paste(dim(output), collapse = "x")))

          optimizer$zero_grad()
          loss <- criterion(output$squeeze(2), y)
          loss$backward()
          optimizer$step()
          total_loss <- total_loss + loss$item()
        })

        cat(sprintf("Epoch %d: Loss: %3f\n", epoch, total_loss / length(dataloader)))
      }

      self$model_path <- paste0("best_model_", tolower(class(self$model)), ".pth")
      torch_save(self$model, self$model_path)
    },

    predict = function(model_path) {
      self$model <- torch_load(model_path)
      self$model$eval()

      dataloader <- self$data_module$train_dataloader()
      predictions <- list()
      labels <- list()

      coro::loop(for (batch in dataloader) {
        output <- self$model(batch[[1]])
        predictions <- c(predictions, output$squeeze(2)$tolist())
        labels <- c(labels, batch[[2]]$tolist())
      })

      return(list(predictions = predictions, labels = labels))
    },

    execute = function(model_path = NULL) {
      if (is.null(model_path)) {
        self$train()
        model_path <- self$model_path
      }
      return(self$predict(model_path = model_path))
    }
  )
)

