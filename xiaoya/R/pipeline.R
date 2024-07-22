library(data.table)
library(torch)
source('R/utils.R')
source("R/data_module.R")
source('R/models.R')

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
        epochs = 20,
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

      # 确保特征数据和target的患者ID一致
      patient_ids <- intersect(unique(self$data_handler$merged_df$PatientID), unique(self$data_handler$target$PatientID))

      # 过滤特征数据和target
      filtered_data <- self$data_handler$merged_df[self$data_handler$merged_df$PatientID %in% patient_ids]
      filtered_target <- self$data_handler$target[self$data_handler$target$PatientID %in% patient_ids]
      features_filtered <- filtered_data[, -c("PatientID", "RecordTime", "Outcome", "LOS"), with = FALSE]
      target_filtered <- filtered_target[, .(PatientID, Outcome)]

      numeric_matrix <- as.matrix(sapply(features_filtered, as.numeric))
      target_tensor <- torch_tensor(as.numeric(target_filtered$Outcome), dtype = torch_float())

      # 按患者ID分组
      patient_data <- split(filtered_data, filtered_data$PatientID)
      patient_target <- split(filtered_target, filtered_target$PatientID) # 分组目标数据

      # 处理缺失值
      patient_matrices <- lapply(patient_data, function(df) {
        df <- df[, -c("PatientID", "RecordTime", "Outcome", "LOS"), with = FALSE]
        df <- as.data.frame(lapply(df, function(col) {
          as.numeric(col)
          col[is.na(col)] <- 0  # 替换缺失值为0
        }))
        as.matrix(df)
      })

      patient_targets <- lapply(patient_target, function(df) {
        outcome <- df$Outcome
        outcome <- as.numeric(outcome)
        return(outcome)
      })

      max_length <- max(sapply(patient_matrices, nrow))

      # 调整每个患者的data和target的长度
      adjusted_matrices <- list()
      adjusted_targets <- list()

      for (i in seq_along(patient_matrices)) {
        data_length <- nrow(patient_matrices[[i]])
        target_length <- length(patient_targets[[i]])

        # 如果data较短，则截断target
        if (data_length < target_length) {
          adjusted_targets[[i]] <- patient_targets[[i]][1:data_length]
        } else {
          adjusted_targets[[i]] <- patient_targets[[i]]
        }

        # 如果target较短，则填充目标为target中的前值
        if (target_length < data_length) {
          padding <- rep(patient_targets[[i]][target_length], data_length - target_length)
          adjusted_targets[[i]] <- c(patient_targets[[i]], padding)
        }

        # 填充data和target到最大长度
        if (nrow(patient_matrices[[i]]) < max_length) {
          data_padding <- matrix(0, nrow = max_length - nrow(patient_matrices[[i]]), ncol = ncol(patient_matrices[[i]]))
          adjusted_data <- rbind(patient_matrices[[i]], data_padding)
        } else {
          adjusted_data <- patient_matrices[[i]]
        }

        if (length(adjusted_targets[[i]]) < max_length) {
          target_padding <- rep(adjusted_targets[[i]][length(adjusted_targets[[i]])], max_length - length(adjusted_targets[[i]]))
          adjusted_targets[[i]] <- c(adjusted_targets[[i]], target_padding)
        }

        adjusted_matrices[[i]] <- torch_tensor(adjusted_data, dtype = torch_float())
        adjusted_targets[[i]] <- torch_tensor(adjusted_targets[[i]], dtype = torch_float())
      }

      # 调整目标的形状以匹配模型的输出形状
      adjusted_targets <- lapply(adjusted_targets, function(t) {
        t <- t$view(c(-1, 1))  # 调整形状为 (batch_size, 1)
        return(t)
      })

      # 构建 EhrDataModule 实例
      self$data_module <- EhrDataModule$new(data = adjusted_matrices, target = adjusted_targets, batch_size = self$config$batch_size)
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

          output <- self$model$forward(x)

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

