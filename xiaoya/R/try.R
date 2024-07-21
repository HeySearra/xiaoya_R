library(h2o)
library(data.table)
Sys.setenv(JAVA_HOME = "D:/Program Files/Java/jdk-11.0.12")

h2o.init()

# 加载数据
merged_df <- fread("datasets/standard_merged_data.csv", colClasses = "numeric")
target <- fread("datasets/standard_target_data.csv", colClasses = "numeric")

merged_df[merged_df == "na"] <- NA
numeric_columns <- names(merged_df)[sapply(merged_df, function(x) all(grepl("^[0-9.]+$", x[!is.na(x)])))]
merged_df[, (numeric_columns) := lapply(.SD, as.numeric), .SDcols = numeric_columns]

merged_df$RecordTime <- as.POSIXct(merged_df$RecordTime, format = "%Y-%m-%d")
target$RecordTime <- as.POSIXct(target$RecordTime, format = "%Y-%m-%d")


# 将 target 数据添加到 merged_df
merged_data <- merge(merged_df, target, by = c("PatientID", "RecordTime"))

data_h2o <- as.h2o(merged_df)

splits <- h2o.splitFrame(data_h2o, ratios = 0.8, seed = 123)
train_h2o <- splits[[1]]
valid_h2o <- splits[[2]]

train_h2o$Outcome <- as.factor(train_h2o$Outcome)
valid_h2o$Outcome <- as.factor(valid_h2o$Outcome)

# 检查训练数据的列数
ncol_train <- ncol(train_h2o)
colnames_train <- colnames(train_h2o)

cat("Number of columns in train_h2o:", ncol_train, "\n")
cat("Column names in train_h2o:", colnames_train, "\n")
str(train_h2o)

  # 定义和训练 GRU 模型
  gru_model <- h2o.deeplearning(
    x = colnames(train_h2o)[-ncol(train_h2o)],  # 特征列
    y = colnames(train_h2o)[ncol(train_h2o)],   # 目标列
    training_frame = train_h2o,
    validation_frame = valid_h2o,
    activation = "RectifierWithDropout",
    hidden = c(50, 50, 50),  # 隐藏层大小
    epochs = 20,
    variable_importances = TRUE
  )


# 模型性能
h2o.performance(gru_model, valid = TRUE)

# 定义和训练 MLP 模型
mlp_model <- h2o.deeplearning(
  x = colnames(train_h2o)[-ncol(train_h2o)],  # 特征列
  y = colnames(train_h2o)[ncol(train_h2o)],  # 目标列
  training_frame = train_h2o,
  validation_frame = valid_h2o,
  activation = "Rectifier",
  hidden = c(50, 50),  # 隐藏层大小
  epochs = 20,
  variable_importances = TRUE
)

# 模型性能
h2o.performance(mlp_model, valid = TRUE)

Pipeline <- R6::R6Class(
  "Pipeline",
  public = list(
    data_handler = NULL,
    model = NULL,
    config = NULL,

    initialize = function(data_handler, model_type, act_function) {
      self$data_handler <- data_handler

      self$config <- list(
        model_type = model_type,
        act_layer = act_function,
        hidden_dim = 50,  # 示例值
        drop = 0.5,       # 示例值
        lr = 0.001        # 示例值
      )

      if (model_type == "GRU") {
        self$model <- h2o.deeplearning(
          x = colnames(data_handler$train_h2o)[-ncol(data_handler$train_h2o)],  # 特征列
          y = colnames(data_handler$train_h2o)[ncol(data_handler$train_h2o)],  # 目标列
          training_frame = data_handler$train_h2o,
          validation_frame = data_handler$valid_h2o,
          activation = "RectifierWithDropout",
          hidden = c(self$config$hidden_dim, self$config$hidden_dim, self$config$hidden_dim),  # 隐藏层大小
          epochs = 20,
          variable_importances = TRUE
        )
      } else if (model_type == "MLP") {
        self$model <- h2o.deeplearning(
          x = colnames(data_handler$train_h2o)[-ncol(data_handler$train_h2o)],  # 特征列
          y = colnames(data_handler$train_h2o)[ncol(data_handler$train_h2o)],  # 目标列
          training_frame = data_handler$train_h2o,
          validation_frame = data_handler$valid_h2o,
          activation = "Rectifier",
          hidden = c(self$config$hidden_dim, self$config$hidden_dim),  # 隐藏层大小
          epochs = 20,
          variable_importances = TRUE
        )
      } else {
        stop("Unsupported model type")
      }
    },

    train = function() {
      # 打印模型性能
      print(h2o.performance(self$model, valid = TRUE))
    }
  )
)

# 示例使用

# 定义 data_handler
data_handler <- list(
  train_h2o = train_h2o,
  valid_h2o = valid_h2o
)

# 创建 GRU 模型并训练
pipeline_gru <- Pipeline$new(data_handler, model_type = "GRU", act_function = "relu")
pipeline_gru$train()

# 创建 MLP 模型并训练
pipeline_mlp <- Pipeline$new(data_handler, model_type = "MLP", act_function = "relu")
pipeline_mlp$train()

h2o.shutdown(prompt = FALSE)

