library(data.table)


# 使用：
# 读取原始数据
# data_analyzer <- DataAnalyzer$new(config, model_path)
# result1 <- data_analyzer$adaptive_feature_importance(df, x, patient_index = 1)


DataAnalyzer <- R6::R6Class(
  "DataAnalyzer",

  public = list(

    config = NULL,
    model_path = NULL,

    initialize = function(config, model_path) {
      self$config <- config
      self$model_path <- model_path
    },

    adaptive_feature_importance = function(df,
                                           x,
                                           patient_index = NULL,
                                           patient_id = NULL) {

      pipeline <- load_pipeline(self$config, self$model_path)
      xid <- if (!is.null(patient_index)) {
        patient_index
      } else {
        get_patient_index(df, patient_id)
      }

      x <- array(x[xid, , ], dim = c(1, dim(x)[-1]))

      pipeline <- adapt_to_pipeline(pipeline, x)

      feat_attn <- pipeline$predict_step(x)

      return(feat_attn[[1]][1,])
    },

    feature_importance = function(df,
                                  x,
                                  patient_index = NULL,
                                  patient_id = NULL) {

      pipeline <- load_pipeline(self$config, self$model_path)
      xid <- if (!is.null(patient_index)) {
        patient_index
      } else {
        get_patient_index(df, patient_id)
      }

      x <- array(x[xid, , ], dim = c(1, dim(x)[-1]))

      pipeline <- adapt_to_pipeline(pipeline, x)

      feat_attn <- pipeline$predict_step(x)

      feat_attn <- feat_attn[[3]][1,]

      column_names <- colnames(df)[7:ncol(df)]

      detail <- lapply(1:length(column_names), function(i) {
        list(
          name = column_names[i],
          value = feat_attn[length(feat_attn), i],
          adaptive = feat_attn[, i]
        )
      })

      detail <- detail[order(sapply(detail, function(x) x$value), decreasing = TRUE)]

      return(detail)
    },

    risk_curve = function(df,
                          x,
                          mean,
                          std,
                          mask = NULL,
                          patient_index = NULL,
                          patient_id = NULL) {

      pipeline <- load_pipeline(self$config, self$model_path)
      if (!is.null(patient_index)) {
        xid <- patient_index
        patient_id <- unique(df$PatientID)[patient_index]
      } else {
        xid <- which(unique(df$PatientID) == patient_id)
      }

      x <- array(x[xid, , ], dim = c(1, dim(x)[-1]))

      pipeline <- adapt_to_pipeline(pipeline, x)

      result <- pipeline$predict_step(x)

      y_hat <- result[[1]][1,]
      feat_attn <- result[[3]][1,]

      column_names <- colnames(df)[7:ncol(df)]
      record_times <- df[df$PatientID == patient_id, "RecordTime"]

      detail <- lapply(1:length(column_names), function(i) {
        list(
          name = column_names[i],
          value = (x[, i] * std[column_names[i]] + mean[column_names[i]]),
          importance = feat_attn[length(feat_attn), i],
          adaptive = feat_attn[, i],
          missing = ifelse(!is.null(mask), mask[, i], NULL),
          unit = ""
        )
      })

      detail <- detail[order(sapply(detail, function(x) x$importance), decreasing = TRUE)]

      return(list(detail = detail, record_times = record_times, y_hat = y_hat))
    },

    ai_advice = function(df,
                         x,
                         mean,
                         std,
                         time_index = -1,
                         patient_index = NULL,
                         patient_id = NULL) {

      pipeline <- load_pipeline(self$config, self$model_path)
      if (!is.null(patient_index)) {
        xid <- patient_index
        patient_id <- unique(df$PatientID)[patient_index]
      } else {
        xid <- which(unique(df$PatientID) == patient_id)
      }

      x <- array(x[xid, , ], dim = c(1, dim(x)[-1]))

      pipeline <- adapt_to_pipeline(pipeline, x)

      result <- pipeline$predict_step(x)

      feat_attn <- result[[3]][1,]

      column_names <- colnames(df)[5:ncol(df)]
      feature_last_step <- feat_attn[time_index, ]
      index_dict <- as.list(setNames(feature_last_step, seq_along(feature_last_step)))
      max_indices <- sort(names(index_dict), index_dict, decreasing = TRUE)[1:3]

      f <- function(x, args) {
        input <- args$input
        i <- args$i
        input[1, , i] <- x
        input <- array(input, dim = c(dim(input)[1], dim(input)[3], dim(input)[2]))
        input <- array_reshape(input, c(dim(input)[1] * dim(input)[2], dim(input)[3]))
        y_hat <- pipeline$predict_step(input)
        return(y_hat[[1]][1, time_index])
      }

      result <- lapply(max_indices, function(i) {
        x0 <- as.numeric(x[1, , i])
        bounds <- c(max(-3, x0 - 1), min(3, x0 + 1))
        args <- list(input = x, i = i)
        res <- optimize(f, interval = bounds, args = args, maximum = TRUE)
        return(list(
          name = column_names[i],
          old_value = x0 * std[column_names[i]] + mean[column_names[i]],
          new_value = res$maximum * std[column_names[i]] + mean[column_names[i]]
        ))
      })

      return(result)
    },

    data_dimension_reduction = function(df,
                                        x,
                                        mean_age,
                                        std_age,
                                        method = "PCA",
                                        dimension = 2,
                                        target = "outcome") {

      num <- length(x)
      patients <- list()
      pid <- unique(df$PatientID)
      record_time <- lapply(split(df$RecordTime, df$PatientID), as.character)

      for (i in 1:num) {
        xi <- array(x[i, , ], dim = c(1, dim(x)[-1]))
        pidi <- pid[i]
        timei <- record_time[[pidi]]

        pipeline <- load_pipeline(self$config, self$model_path)
        pipeline <- pipeline$load_from_checkpoint(self$model_path)

        xi <- array_reshape(c(xi, xi), c(1, dim(xi)[2], dim(xi)[3]))
        if (pipeline$on_gpu) {
          xi <- xi %>% torch::to_device(device = torch::cuda_device(0))
        }

        result <- pipeline$predict_step(xi)

        embedding <- result[[2]][1, , ] %>% torch::cpu() %>% torch::detach() %>% torch::numpy() %>% as.vector()
        y_hat <- result[[1]][1, , ] %>% torch::cpu() %>% torch::detach() %>% torch::numpy() %>% as.vector()

        if (method == "PCA") {
          reduction_model <- prcomp(embedding)$x
        } else if (method == "TSNE") {
          reduction_model <- Rtsne::Rtsne(embedding, dims = dimension, verbose = TRUE)$Y
        }

        if (length(reduction_model[, 1]) != length(reduction_model[1, ])) {
          next
        }

        if (target == "outcome") {
          y_hat <- y_hat[, 1]
        } else {
          y_hat <- y_hat[, 2]
        }

        patient <- list()
        if (dimension == 2) {
          patient$data <- data.frame(reduction_model[, 1], reduction_model[, 2], y_hat)
        } else if (dimension == 3) {
          patient$data <- data.frame(reduction_model[, 1], reduction_model[, 2], reduction_model[, 3], y_hat)
        }
        patient$patient_id <- pidi
        patient$record_time <- timei
        if (!is.null(std_age) && !is.null(mean_age)) {
          patient$age <- as.integer(xi[1, 1, 2] * std_age + mean_age)
        }
        patients[[i]] <- patient
      }

      return(patients)
    },

    similar_patients = function(df,
                                x,
                                mean,
                                std,
                                patient_index = NULL,
                                patient_id = NULL,
                                n_clu = 10,
                                topk = 6) {

      pipeline <- load_pipeline(self$config, self$model_path)
      if (!is.null(patient_index)) {
        xid <- patient_index
        patient_id <- unique(df$PatientID)[patient_index]
      } else {
        xid <- which(unique(df$PatientID) == patient_id)
      }

      x <- array(x[xid, , ], dim = c(1, dim(x)[-1]))

      pipeline <- adapt_to_pipeline(pipeline, x)

      result <- pipeline$predict_step(x)

      feat_attn <- result[[3]][1, ]

      column_names <- colnames(df)[5:ncol(df)]
      feature_last_step <- feat_attn[time_index, ]
      index_dict <- as.list(setNames(feature_last_step, seq_along(feature_last_step)))
      max_indices <- sort(names(index_dict), index_dict, decreasing = TRUE)[1:3]

      f <- function(x, args) {
        input <- args$input
        i <- args$i
        input[1, , i] <- x
        input <- array(input, dim = c(dim(input)[1], dim(input)[3], dim(input)[2]))
        input <- array_reshape(input, c(dim(input)[1] * dim(input)[2], dim(input)[3]))
        y_hat <- pipeline$predict_step(input)
        return(y_hat[[1]][1, time_index])
      }

      result <- lapply(max_indices, function(i) {
        x0 <- as.numeric(x[1, , i])
        bounds <- c(max(-3, x0 - 1), min(3, x0 + 1))
        args <- list(input = x, i = i)
        res <- optimize(f, interval = bounds, args = args, maximum = TRUE)
        return(list(
          name = column_names[i],
          old_value = x0 * std[column_names[i]] + mean[column_names[i]],
          new_value = res$maximum * std[column_names[i]] + mean[column_names[i]]
        ))
      })

      return(result)
    },

    analyze_dataset = function(df,
                               x,
                               feature,
                               mean,
                               std) {

      pipeline <- load_pipeline(self$config, self$model_path)
      labtest_feature_index <- which(colnames(df) %in% feature) - 6
      lens <- sapply(x, length)

      x <- lapply(x, function(item) { as.array(item) })
      x <- do.call(rbind, x)
      x <- array_reshape(x, c(length(x), dim(x)[2], dim(x)[3]))
      x <- aperm(x, c(1, 3, 2))
      if (pipeline$on_gpu) {
        x <- x %>% torch::to_device(device = torch::cuda_device(0))
      }

      result <- pipeline$predict_step(x)

      y_hat <- result[[1]] %>% torch::cpu() %>% torch::detach() %>% torch::numpy() %>% as.array()
      feat_attn <- result[[3]] %>% torch::cpu() %>% torch::detach() %>% torch::numpy() %>% as.array()

      x <- x[, , 2 + labtest_feature_index] %>% torch::cpu() %>% torch::detach() %>% torch::numpy() %>% as.array()

      data <- data.frame(
        Value = x * std[[feature]] + mean[[feature]],
        Attention = feat_attn * 100,
        Outcome = y_hat[, , 1]
      )

      outcome_bins <- seq(0, 1, by = 0.2)
      attn_bins <- seq(0, 100, by = 1)
      value_bins <- seq(min(data$Value), max(data$Value), length.out = 52)

      data_bar_2D <- list()
      data_line_2D <- list()
      data_3D <- list()

      for (outcome in split(data, cut(data$Outcome, breaks = outcome_bins, include.lowest = TRUE))) {
        data_2D_outcome <- list()
        data_3D_outcome <- list()
        for (i in 1:(length(value_bins) - 1)) {
          by_value <- outcome[outcome$Value >= value_bins[i] & outcome$Value < value_bins[i + 1], ]
          data_2D_outcome[[i]] <- c(value_bins[i + 1], nrow(by_value))
          for (j in 1:(length(attn_bins) - 1)) {
            by_attn <- by_value[by_value$Attention >= attn_bins[j] & by_value$Attention < attn_bins[j + 1], ]
            data_3D_outcome[[j]] <- c(value_bins[i + 1], attn_bins[j + 1], nrow(by_attn))
          }
        }
        data_bar_2D <- append(data_bar_2D, list(data_2D_outcome))
        data_3D <- append(data_3D, list(data_3D_outcome))
      }

      for (i in 1:(length(value_bins) - 1)) {
        by_value <- data[data$Value >= value_bins[i] & data$Value < value_bins[i + 1], ]
        data_line_2D[[i]] <- c(value_bins[i + 1], mean(by_value$Attention))
      }

      return(list(data_bar_2D, data_line_2D, data_3D))
    }


  )
)
