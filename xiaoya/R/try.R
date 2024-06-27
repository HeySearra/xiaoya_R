library(dplyr)
library(palmerpenguins)

penguins %>% glimpse()
library(torch)
torch_tensor(1)
penguins_dataset <- dataset(

  name = "penguins_dataset",

  initialize = function(df) {

    df <- na.omit(df)

    # continuous input data (x_cont)
    x_cont <- df[ , c("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "year")] %>%
      as.matrix()
    self$x_cont <- torch_tensor(x_cont)

    # categorical input data (x_cat)
    x_cat <- df[ , c("island", "sex")]
    x_cat$island <- as.integer(x_cat$island)
    x_cat$sex <- as.integer(x_cat$sex)
    self$x_cat <- as.matrix(x_cat) %>% torch_tensor()

    # target data (y)
    species <- as.integer(df$species)
    self$y <- torch_tensor(species)

  },

  .getitem = function(i) {
    list(x_cont = self$x_cont[i, ], x_cat = self$x_cat[i, ], y = self$y[i])

  },

  .length = function() {
    self$y$size()[[1]]
  }

)

train_indices <- sample(1:nrow(penguins), 250)

train_ds <- penguins_dataset(penguins[train_indices, ])
valid_ds <- penguins_dataset(penguins[setdiff(1:nrow(penguins), train_indices), ])

length(train_ds)
length(valid_ds)


train_dl <- train_ds$dataloader(batch_size = 16, shuffle = TRUE)

valid_dl <- valid_ds %>% dataloader(batch_size = 16, shuffle = FALSE)
