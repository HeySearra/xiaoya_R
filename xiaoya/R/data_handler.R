library(data.table)
library(zoo)

# 使用：
# labtest_data <- fread('datasets/raw_labtest_data.csv')
# events_data <- fread('datasets/raw_events_data.csv')
# target_data <- fread('datasets/raw_target_data.csv')

# data_handler <- init_data_handler(labtest_data, events_data, target_data, data_path)
# data_handler <- format_and_merge_dataframes(data_handler)
# data_handler <- save_processed_data(data_handler)
# result <- analyze_dataset(data_handler)
# data_handler <- split_dataset(data_handler, train_size = 70, val_size = 10, test_size = 20, seed = 42)


init_data_handler <- function(labtest_data, events_data, target_data, data_path) {

  data_handler <- list(
    raw_df = list(labtest = as.data.table(labtest_data),
                  events = as.data.table(events_data),
                  target = as.data.table(target_data)),
    standard_df = list(),
    merged_df = NULL,
    data_path = data_path
  )

  return(data_handler)
}

format_dataframe <- function(data_handler, format) {
  stopifnot(format %in% c("labtest", "events", "target"))

  df <- copy(data_handler$raw_df[[format]])

  if (format == "target") {
    df <- unique(df, by = c("PatientID", "RecordTime"))
  } else {
    df <- unique(df, by = c("PatientID", "RecordTime", "Name"))

    names <- unique(na.omit(df[, df$Name]))

    df_new <- data.table(PatientID = integer(), RecordTime = integer())
    for (name in names) {
      df_new[[name]] <- numeric()
    }

    setkey(df, PatientID, RecordTime)
    for (patient_id in unique(df[, PatientID])) {
      for (record_time in unique(df[PatientID == patient_id, RecordTime])) {
        row <- df[PatientID == patient_id & RecordTime == record_time, .(Name, Value)]
        for (i in seq_len(nrow(row))) {
          if (nrow(df_new[PatientID == patient_id & RecordTime == record_time]) == 0) {
            # 创建新行并添加到 df_new 中
            new_row <- data.table(PatientID = patient_id, RecordTime = record_time)
            column_name <- row$Name[i]
            column_value <- row$Value[i]
            new_row[, (column_name) := column_value]
            df_new <- rbindlist(list(df_new, new_row), use.names = TRUE, fill = TRUE)
          }
          else {
            # 更新现有行中的值
            df_new[PatientID == patient_id & RecordTime == record_time, (row$Name[i]) := row$Value[i]]
          }
        }
      }
    }

    df <- df_new
  }

  df[, RecordTime := as.POSIXct(RecordTime, format = "%Y/%m/%d")]
  setkey(df, PatientID, RecordTime)
  data_handler$standard_df[[format]] <- df
  return(data_handler)
}

merge_dataframes <- function(data_handler) {
  stopifnot(!is.null(data_handler$standard_df$labtest))
  stopifnot(!is.null(data_handler$standard_df$events))
  stopifnot(!is.null(data_handler$standard_df$target))

  labtest_standard_df <- data_handler$standard_df$labtest
  events_standard_df <- data_handler$standard_df$events
  target_standard_df <- data_handler$standard_df$target

  df <- merge(labtest_standard_df, events_standard_df, by = c("PatientID", "RecordTime"), all = TRUE)
  df <- merge(df, target_standard_df, by = c("PatientID", "RecordTime"), all = TRUE)

  event_cols <- c("Sex")
  df[, (event_cols) := lapply(.SD, function(x) zoo::na.locf(x, na.rm = FALSE)), .SDcols = event_cols, by = PatientID]

  cols <- c("PatientID", "RecordTime", "Outcome", "LOS", "Sex", "Age")

  all_cols <- names(df)
  cols <- cols[cols %in% all_cols]  # 仅保留存在于df中的列
  all_cols <- c(cols, setdiff(all_cols, cols))
  df <- df[, ..all_cols]

  # 保存结果到 data_handler
  data_handler$merged_df <- df

  return(data_handler)
}


format_and_merge_dataframes <- function(data_handler) {
  data_handler <- format_dataframe(data_handler, 'labtest')
  data_handler <- format_dataframe(data_handler, 'events')
  data_handler <- format_dataframe(data_handler, 'target')

  data_handler <- merge_dataframes(data_handler)

  return(data_handler)
}

save_processed_data <- function(data_handler) {

  fwrite(data_handler$standard_df$labtest,
         file.path(data_handler$data_path, 'standard_labtest_data.csv'),
         row.names = FALSE, na = "na")

  fwrite(data_handler$standard_df$events,
         file.path(data_handler$data_path, 'standard_events_data.csv'),
         row.names = FALSE, na = "na")

  fwrite(data_handler$standard_df$target,
         file.path(data_handler$data_path, 'standard_target_data.csv'),
         row.names = FALSE, na = "na")

  fwrite(data_handler$merged_df,
         file.path(data_handler$data_path, 'standard_merged_data.csv'),
         row.names = FALSE, na = "na")

  return(data_handler)
}

extract_features <- function(data_handler, format) {
  stopifnot(format %in% c("labtest", "events", "target"))

  df <- data_handler$raw_df[[format]]

  if (format %in% c("labtest", "events")) {
    feats <- unique(na.omit(df[, Name]))
  } else {
    feats <- setdiff(names(df), c("PatientID", "RecordTime"))
  }

  data_handler$raw_features[[format]] <- feats
  return(data_handler)
}

list_all_features <- function(data_handler) {
  formats <- c("labtest", "events", "target")
  for(format in formats){
    data_handler <- extract_features(data_handler, format)
  }

  return(data_handler)
}
analyze_dataset <- function(data_handler) {
  detail <- list()

  features <- names(data_handler$merged_df)
  features <- setdiff(features, "RecordTime")  # 删除 "RecordTime"

  for (feature in features) {
    info <- list()
    info$name <- feature
    if (is.list(data_handler$merged_df[[feature]])) {
      # 如果特征是一个列表，可以选择一个元素进行分析，比如第一个元素
      values <- unlist(data_handler$merged_df[[feature]][[1]])
    } else {
      values <- data_handler$merged_df[[feature]]
    }
    info$stats <- list(
      list(name = "count", value = sum(!is.na(values))),
      list(name = "missing", value = paste0(round(100 * mean(is.na(values)), 2), "%")),
      list(name = "mean", value = round(mean(values, na.rm = TRUE), 2)),
      list(name = "max", value = round(max(values, na.rm = TRUE), 2)),
      list(name = "min", value = round(min(values, na.rm = TRUE), 2)),
      list(name = "median", value = round(median(values, na.rm = TRUE), 2)),
      list(name = "std", value = round(sd(values, na.rm = TRUE), 2))
    )
    detail <- append(detail, list(info))
  }

  # 返回包含所有特征信息的列表
  return(list(detail = detail))
}


split_dataset <- function(data_handler, train_size = 70, val_size = 10, test_size = 20, seed = 42){

  stopifnot(train_size + val_size + test_size == 100)

  # 按照病人ID对数据集进行分组
  grouped <- split(data_handler$merged_df, data_handler$merged_df$PatientID)
  patients <- names(grouped)

  # 获得病人疾病结果
  patients_outcome <- sapply(patients, function(patient_id) grouped[[patient_id]]$Outcome[1])

  # 获取训练和验证/测试病人的ID
  set.seed(seed)
  train_val_patients <- patients[sample(1:length(patients),
                                        size = round(train_size/100 * length(patients)),
                                        replace = FALSE,
                                        prob = patients_outcome)]
  test_patients <- setdiff(patients, train_val_patients)

  # 从训练和验证数据中再获取训练数据和验证数据
  train_val_patients_outcome <- sapply(train_val_patients, function(patient_id) grouped[[patient_id]]$Outcome[1])

  train_patients <- train_val_patients[sample(1:length(train_val_patients),
                                              size = round(train_size/(train_size + val_size) * length(train_val_patients)),
                                              replace = FALSE,
                                              prob = train_val_patients_outcome)]
  val_patients <- setdiff(train_val_patients, train_patients)

  data_handler$split_datasets <- list(train = do.call(rbind, grouped[train_patients]),
                                      val   = do.call(rbind, grouped[val_patients]),
                                      test  = do.call(rbind, grouped[test_patients]))

  return(data_handler)
}

