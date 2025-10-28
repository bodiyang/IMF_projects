library(httr)
library(readxl)
library(dplyr)
library(forecast)
library(ggplot2)

# Retrieve currency in circulation data from the Bank of Zambia website
url <- "https://www.boz.zm/FORTNIGHTLYTIMESERIESendingSeptember202024.xlsx"
GET(url, write_disk(tf <- tempfile(fileext = ".xlsx")))
df_raw <- read_excel(tf, sheet = "Reserve Money", skip = 6)

# Data Cleaning
circulation <- as.numeric(df_raw$Circulation[1:2965])
circulation_ts <- ts(circulation)

# ETS Model fit
fit <- ets(circulation_ts, model = "MMN")

# Summary
summary(fit)

# Plot
autoplot(fit) +
  ggtitle("ETS Model Fit to Circulation Data") +
  xlab("Date") + ylab("Circulation") +
  theme_minimal()

# Forecast for next 7 days
forecast_values <- forecast(fit, h = 7)
print(forecast_values)

