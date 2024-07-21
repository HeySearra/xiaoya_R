library(h2o)
library(data.table)
library(R6)
Sys.setenv(JAVA_HOME = "D:/Program Files/Java/jdk-11.0.12")


EHRDataset <- R6Class(
  "EHRDataset",
  public = list(
    merged_df = NULL,
    target = NULL,
    data_handler = NULL,

    initialize = function(merged_path, target_path) {
      h2o.init()
      # 加载数据
      self$merged_df <- fread(merged_path)
      self$target <- fread(target_path)

      # 处理缺失值
      self$merged_df[self$merged_df == "na"] <- NA

      # 转换数值列为数值类型
      numeric_columns <- names(self$merged_df)[sapply(self$merged_df, function(x) all(grepl("^[0-9.]+$", x[!is.na(x)])))]
      self$merged_df[, (numeric_columns) := lapply(.SD, as.numeric), .SDcols = numeric_columns]

      # 处理日期列
      self$merged_df$RecordTime <- as.Date(self$merged_df$RecordTime, format = "%Y-%m-%d")
      self$target$RecordTime <- as.Date(self$target$RecordTime, format = "%Y-%m-%d")

      # 将 target 数据添加到 merged_df
      merged_data <- merge(self$merged_df, self$target, by = c("PatientID", "RecordTime", "Outcome", "LOS"))

      # 将数据转换为 H2O 格式
      data_h2o <- as.h2o(merged_data)

      # 分割数据
      splits <- h2o.splitFrame(data_h2o, ratios = 0.8, seed = 123)
      train_h2o <<- splits[[1]]
      valid_h2o <<- splits[[2]]

      # 转换目标变量为因子类型
      train_h2o$Outcome <- as.factor(train_h2o$Outcome)
      valid_h2o$Outcome <- as.factor(valid_h2o$Outcome)

      # 创建 data_handler
      self$data_handler <- list(
        train_h2o = train_h2o,
        valid_h2o = valid_h2o
      )

    },

    get_data_handler = function() {
      return(self$data_handler)
    }
  )
)
