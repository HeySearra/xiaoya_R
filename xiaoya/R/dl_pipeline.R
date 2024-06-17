library(torch)
library(data.table)
library(R6)

# Define the DlPipeline class
DlPipeline <- R6Class("DlPipeline",
                      public = list(
                        config = NULL,
                        demo_dim = 0,
                        lab_dim = 0,
                        input_dim = 0,
                        hidden_dim = 32,
                        output_dim = 1,
                        learning_rate = 1e-3,
                        task = "multitask",
                        los_info = NULL,
                        model_name = "GRU",
                        main_metric = "auprc",
                        time_aware = FALSE,
                        embedding = NULL,
                        ehr_encoder = NULL,
                        head = NULL,
                        cur_best_performance = NULL,
                        validation_step_outputs = NULL,
                        test_step_outputs = NULL,
                        test_performance = NULL,
                        test_outputs = NULL,

                        initialize = function(config) {
                          self$config <- config
                          self$demo_dim <- ifelse("demo_dim" %in% names(config), config$demo_dim, 0)
                          self$lab_dim <- ifelse("lab_dim" %in% names(config), config$lab_dim, 0)
                          self$input_dim <- self$demo_dim + self$lab_dim
                          config$input_dim <- self$input_dim
                          self$hidden_dim <- ifelse("hidden_dim" %in% names(config), config$hidden_dim, 32)
                          self$output_dim <- ifelse("output_dim" %in% names(config), config$output_dim, 1)
                          self$learning_rate <- ifelse("learning_rate" %in% names(config), config$learning_rate, 1e-3)
                          self$task <- ifelse("task" %in% names(config), config$task, "multitask")
                          self$los_info <- ifelse("los_info" %in% names(config), config$los_info, NULL)
                          self$model_name <- ifelse("model" %in% names(config), config$model, "GRU")
                          self$main_metric <- ifelse("main_metric" %in% names(config), config$main_metric, "auprc")
                          self$time_aware <- ifelse("time_aware" %in% names(config), config$time_aware, FALSE)

                          if (self$model_name == "StageNet") {
                            config$chunk_size <- self$hidden_dim
                          }

                          model_class <- get(paste0("models::", self$model_name))
                          self$ehr_encoder <- do.call(model_class$new, config)

                          if (self$task == "outcome") {
                            self$head <- nn_sequential(nn_linear(self$hidden_dim, self$output_dim), nn_dropout(0.0), nn_sigmoid())
                          } else if (self$task == "los") {
                            self$head <- nn_sequential(nn_linear(self$hidden_dim, self$output_dim), nn_dropout(0.0))
                          } else if (self$task == "multitask") {
                            self$head <- models::heads$MultitaskHead$new(self$hidden_dim, self$output_dim, drop = 0.0)
                          }

                          self$cur_best_performance <- list()
                          self$validation_step_outputs <- list()
                          self$test_step_outputs <- list()
                          self$test_performance <- list()
                          self$test_outputs <- list()
                        },

                        forward = function(x, lens) {
                          if (self$model_name %in% c("ConCare")) {
                            x_demo <- x[, 1, 1:self$demo_dim]
                            x_lab <- x[, , (self$demo_dim + 1):ncol(x)]
                            mask <- generate_mask(lens)
                            embedding <- self$ehr_encoder(x_lab, x_demo, mask)
                            y_hat <- self$head(embedding)
                            return(list(y_hat, embedding, mask))
                          } else if (self$model_name %in% c("AdaCare")) {
                            mask <- generate_mask(lens)
                            embedding <- self$ehr_encoder(x, mask)
                            y_hat <- self$head(embedding)
                            return(list(y_hat, embedding, mask))
                          } else {
                            # Handle other models
                          }
                        },

                        get_loss = function(x, y, lens) {
                          if (self$model_name %in% c("ConCare")) {
                            result <- self$forward(x, lens)
                            y_hat <- result[[1]]
                            embedding <- result[[2]]
                            mask <- result[[3]]
                            loss <- get_simple_loss(y_hat, y, self$task)
                          } else if (self$model_name %in% c("AdaCare")) {
                            result <- self$forward(x, lens)
                            y_hat <- result[[1]]
                            embedding <- result[[2]]
                            mask <- result[[3]]
                            loss <- get_simple_loss(y_hat, y, self$task)
                          } else {
                            # Handle other models
                          }
                          return(list(loss, y, y_hat))
                        },

                        training_step = function(batch, batch_idx) {
                          x <- batch[[1]]
                          y <- batch[[2]]
                          lens <- batch[[3]]
                          pid <- batch[[4]]
                          result <- self$get_loss(x, y, lens)
                          loss <- result[[1]]
                          y <- result[[2]]
                          y_hat <- result[[3]]
                          self$log("train_loss", loss)
                          return(loss)
                        },

                        validation_step = function(batch, batch_idx) {
                          x <- batch[[1]]
                          y <- batch[[2]]
                          lens <- batch[[3]]
                          pid <- batch[[4]]
                          result <- self$get_loss(x, y, lens)
                          loss <- result[[1]]
                          y <- result[[2]]
                          y_hat <- result[[3]]
                          self$log("val_loss", loss)
                          outs <- list(y_pred = y_hat, y_true = y, val_loss = loss)
                          self$validation_step_outputs <- append(self$validation_step_outputs, list(outs))
                          return(loss)
                        },

                        on_validation_epoch_end = function() {
                          y_pred <- torch_cat(lapply(self$validation_step_outputs, function(x) x$y_pred))
                          y_true <- torch_cat(lapply(self$validation_step_outputs, function(x) x$y_true))
                          loss <- mean(torch_stack(lapply(self$validation_step_outputs, function(x) x$val_loss)))
                          self$log("val_loss_epoch", loss)
                          metrics <- get_all_metrics(y_pred, y_true, self$task, self$los_info)
                          lapply(names(metrics), function(k) self$log(k, metrics[[k]]))
                          main_score <- metrics[[self$main_metric]]
                          if (check_metric_is_better(self$cur_best_performance, self$main_metric, main_score, self$task)) {
                            self$cur_best_performance <- metrics
                            lapply(names(metrics), function(k) self$log(paste0("best_", k), metrics[[k]]))
                          }
                          self$validation_step_outputs <- list()
                          return(main_score)
                        },

                        test_step = function(batch, batch_idx) {
                          x <- batch[[1]]
                          y <- batch[[2]]
                          lens <- batch[[3]]
                          pid <- batch[[4]]
                          result <- self$get_loss(x, y, lens)
                          loss <- result[[1]]
                          y <- result[[2]]
                          y_hat <- result[[3]]
                          outs <- list(y_pred = y_hat, y_true = y, lens = lens)
                          self$test_step_outputs <- append(self$test_step_outputs, list(outs))
                          return(loss)
                        },

                        on_test_epoch_end = function() {
                          y_pred <- torch_cat(lapply(self$test_step_outputs, function(x) x$y_pred))
                          y_true <- torch_cat(lapply(self$test_step_outputs, function(x) x$y_true))
                          lens <- torch_cat(lapply(self$test_step_outputs, function(x) x$lens))
                          self$test_performance <- get_all_metrics(y_pred, y_true, self$task, self$los_info)
                          self$test_outputs <- list(preds = y_pred, labels = y_true, lens = lens)
                          self$test_step_outputs <- list()
                          return(self$test_performance)
                        },

                        predict_step = function(x) {
                          lens <- torch_tensor(sapply(1:nrow(x), function(i) nrow(x[i, , ])))
                          if (self$model_name %in% c("ConCare")) {
                            result <- self$forward(x, lens)
                            y_hat <- result[[1]]
                            embedding <- result[[2]]
                            mask <- result[[3]]
                            return(list(y_hat, embedding, mask))
                          } else if (self$model_name %in% c("AdaCare")) {
                            result <- self$forward(x, lens)
                            y_hat <- result[[1]]
                            embedding <- result[[2]]
                            mask <- result[[3]]
                            return(list(y_hat, embedding, mask))
                          } else {
                            # Handle other models
                          }
                        },

                        configure_optimizers = function() {
                          optimizer <- optim_adamw(self$parameters(), lr = self$learning_rate)
                          return(optimizer)
                        }
                      )
)

# Helper functions (placeholders for actual implementations)
generate_mask <- function(lens) {
  # Placeholder for generate_mask function
}

get_simple_loss <- function(y_hat, y, task) {
  # Placeholder for get_simple_loss function
}

get_all_metrics <- function(y_pred, y_true, task, los_info) {
  # Placeholder for get_all_metrics function
}

check_metric_is_better <- function(cur_best_performance, main_metric, main_score, task) {
  # Placeholder for check_metric_is_better function
}

# Assuming models is a namespace for different model classes and heads
models <- list(
  ConCare = ConCareClass, # Placeholder for the actual ConCare class
  AdaCare = AdaCareClass, # Placeholder for the actual AdaCare class
  heads = list(
    MultitaskHead = MultitaskHeadClass # Placeholder for the actual MultitaskHead class
  )
)
