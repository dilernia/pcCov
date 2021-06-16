# pcCov

Inference of Partial Correlations for Multivariate Time Series

## Description

This R package implements methods for inference of partial correlations for a stationary and ergodic Gaussian process. Methods of inference are based on second-order Taylor Series approximation of the covariances of the partial correlations.

</p>

# Table of contents

  - [Overview of main functions](#overview-main)
  - [Installation](#install)
  - [Examples](#examples)
  - [References](#refs)

<h2 id="overview-main">

Overview of main functions

</h2>

<table>
<colgroup>
<col style="width: 28%" />
<col style="width: 71%" />
</colgroup>
<thead>
<tr class="header">
<th>Function</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr class="even">
<td><code>partialCov</code></td>
<td>Calculates a second-order Taylor Series estimate of the covariance matrix for partial correlations of a stationary Gaussian process.</td>
</tr>
<tr class="odd">
<td><code>bootVar</code></td>
<td>Calculates block-bootstrap covariance matrix estimates and confidence intervals for partial correlations of a multivariate time series.</td>
</tr>
<tr class="even">
<td><code>royVar</code></td>
<td>Calculates Roy (1989)'s asymptotic covariance matrix for marginal or partial correlations.</td>
</tr>
<tr class="odd">
<td><code>royVarhelper</code></td>
<td>Calculates matrix of indices for assisting in efficient calculation of asymptotic covariance matrix estimate using either royVar_cpp() or partialCov_cpp()</td>
</tr>
<tr class="even">
<td><code>vcm_cpp</code></td>
<td>Implements the a variance components model proposed by Fiecas et al. (2017).</td>
</tr>
<tr class="odd">
<td><code>varSim</code></td>
<td>Simulates data from a mean 0 first-order vector auto-regressive (VAR) model.</td>
</tr>
<tr class="even">
<td><code>corrMat_cpp</code></td>
<td>Calculates marginal or partial correlation matrix.</td>
</tr>
<tr class="odd">
<td><code>varSim</code></td>
<td>Simulates data from a mean 0 first-order vector auto-regressive (VAR) model.</td>
</tr>
<tr class="even">
<td><code>invCov2part_cpp</code></td>
<td>Calculates partial correlation matrix from the inverse-covariance matrix.</td>
</tr>
<tr class="odd">
<td><code>bdiagArray_cpp</code></td>
<td>Constructs a block-diagonal matrix from a 3D array.</td>
</tr>
<tr class="even">
<td><code>eigenMult2</code></td>
<td>Efficiently multiplies two matrices. Similar functions are available for 3 (`eigenMult3`) and 4 (`eigenMult4`) matrices.</td>
</tr>
</tbody>
</table>

<h2 id="install">

Installation

</h2>

Install the latest version of the package from GitHub with the following R code:

    if("devtools" %in% installed.packages() == FALSE) {
        install.packages("devtools")
    }
    
    # Installing from GitHub
    devtools::install_github("dilernia/pcCov")

<h2 id="examples">

Examples

</h2>

Here we will walk through brief examples of using some key functions.

### Inference of correlations for time series data

First, we simulate data from a first-order autoregressive (AR(1)) model for demonstration
purposes.

``` r
library(tidyverse)
library(stepDetect)

# Display help files
?featureEngineer
?machineLearn

# Simulating activity data set
set.seed(1994)
tsLength <- 200
aTime <- Sys.time()
simData <- data.frame(start_time = round(seq(from = aTime, to = aTime + tsLength*60 - 1, by = 60), "mins"),
                      end_time = round(seq(from = aTime, to = aTime + tsLength*60 - 1, by = 60), "mins") + 60,
                      steps = as.numeric(rpois(n = tsLength, lambda = 30)*rbinom(n = tsLength, size = 1, prob = 0.80)))

# Artificially creating true labels
simData$Activity <- sapply(simData$steps / max(simData$steps), FUN = function(x) {
  sample(c("Walking", "Other"), size = 1, prob = c(x, 1 - x))
})
```

We can visualize this simulated data set using ggplot:

``` r
library(ggridges)

# Visualizing artificial step count data
ggplot(simData, aes(x = start_time, y = steps)) +
    geom_point() + labs(title = "Simulated Step Count Data", x = "", y = "Steps per Minute") + theme_bw() +
    theme(legend.position = "none")
```

``` r
# Visualizing accelerometer data
simData %>% ggplot(aes(y = Activity, x = steps)) +
  geom_density_ridges(alpha = 0.8) +
  labs(y = "Activity", x = "", title = "Step Counts by Activity") + 
  theme_bw() + theme(axis.text.x = element_blank(),
                     axis.ticks.x = element_blank())
```

    ## Picking joint bandwidth of 3.44

Next, let’s construct some features for fitting our machine learning
models based on the univariate step counts across time. We consider 7
different time-domain functions of rolling windows of the step counts,
and two different window-lengths in total. We also are constructing 8
different frequency domain features (in this case discrete wavelet
features) for our machine learning models as well.

``` r
# Functions to apply to rolling window
myFuns <- setNames(list(mean, max, min, sd, IQR,
                        PerformanceAnalytics::skewness,
                        function(x, na.rm){sum(x * 1:length(x)) / sum((1:length(x))^2)}),
                   c("mean", "max", "min", "sd", "iqr", "skew", "slope"))

# Creating new features
designData <- featureEngineer(steps = simData$steps, funs = myFuns, winLengths = c(3, 4), waves = 8) %>%
  dplyr::mutate(Activity = simData$Activity)
```

We can then visualize some of our newly constructed features.

``` r
# Visualizing rolling statistics
winl <- 3
designData %>% pivot_longer(cols = contains(paste0("_", winl)), names_to = "variable", values_to = "value") %>%
  ggplot(aes(y = Activity, x = steps)) +
  geom_density_ridges(alpha = 0.8) +
  facet_grid(cols = vars(variable), scales = "free_y") +
  labs(y = "Activity", x = "", title = "Rolling Statistics by Activity",
       subtitle = paste0("Window Length of ", winl, " Minutes")) + 
  theme_bw() + theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
```

    ## Picking joint bandwidth of 3.44
    ## Picking joint bandwidth of 3.44
    ## Picking joint bandwidth of 3.44
    ## Picking joint bandwidth of 3.44
    ## Picking joint bandwidth of 3.44
    ## Picking joint bandwidth of 3.44
    ## Picking joint bandwidth of 3.44

Then, using our constructed features we fit several different models: a
first-order elastic net penalized logistic regression model, a
second-order elastic net penalized logistic regression model, and a
regularized random forest model. For each of these models, we can
implement a Synthetic Minority Over-sampling TEchnique (SMOTE) or
reweighting of observations for improved sensitivity of detecting
exercise sessions when class imbalance in the data is present.

``` r
# Instantiating list for model results
myMods <- list()

# Unadjusted

# Fitting 1st-order elastic net model
myMods$modelRes1 <- machineLearn(designData = designData, model = "glmnet",
                         nfolds = 5, secOrder = FALSE, ncores = 1, smote = FALSE)

# Fitting 2nd-order elastic net model
myMods$modelRes2 <- machineLearn(designData = designData, model = "glmnet",
                         nfolds = 5, secOrder = TRUE, ncores = 1, smote = FALSE)

# Fitting 1st-order elastic net model
myMods$modelResRRF <- machineLearn(designData = designData, model = "RRF",
                         nfolds = 5, secOrder = FALSE, ncores = 1, smote = FALSE)
```

    ## Registered S3 method overwritten by 'RRF':
    ##   method      from        
    ##   plot.margin randomForest

``` r
# SMOTE

# Fitting 1st-order elastic net model w/ SMOTE
myMods$modelResSMOTE1 <- machineLearn(designData = designData, model = "glmnet",
                         nfolds = 5, secOrder = FALSE, ncores = 1, smote = TRUE)

# Fitting 2nd-order elastic net model w/ SMOTE
myMods$modelResSMOTE2 <- machineLearn(designData = designData, model = "glmnet",
                         nfolds = 5, secOrder = TRUE, ncores = 1, smote = TRUE)

# Fitting 1st-order elastic net model w/ SMOTE
myMods$modelResSMOTERRF <- machineLearn(designData = designData, model = "RRF",
                         nfolds = 5, secOrder = FALSE, ncores = 1, smote = TRUE)

# Reweighting

# Fitting 1st-order elastic net model
myMods$modelResRW1 <- machineLearn(designData = designData, model = "glmnet", reweighted = TRUE,
                         nfolds = 5, secOrder = FALSE, ncores = 1, smote = FALSE)

# Fitting 2nd-order elastic net model
myMods$modelResRW2 <- machineLearn(designData = designData, model = "glmnet", reweighted = TRUE,
                         nfolds = 5, secOrder = TRUE, ncores = 1, smote = FALSE)

# Fitting 1st-order elastic net model
myMods$modelResRWRRF <- machineLearn(designData = designData, model = "RRF", reweighted = TRUE,
                         nfolds = 5, secOrder = FALSE, ncores = 1, smote = FALSE)
```

Let’s look at the most important features for each model by creating a
grid of feature importance plots using the `importancePlot()` function.
The first column contains the 1st-order elastic net models, the second
column the 2nd-order elastic net models, and the third column the random
forest models. The first row contains the unadjusted models, the second
row the SMOTE models, and the third row the reweighted models.

``` r
lapply(1:3, FUN = function(m) {
    lapply(myMods[(1 + (m-1)*3):(3*m)], FUN = function(modelRes){
      modelRes[[1]] %>% importancePlot() %>% ggplot2::ggplotGrob()}) %>% 
        do.call(what = 'cbind')}) %>% do.call(what = 'rbind') %>% grid::grid.draw()
```

Now let’s look at the confusion matrix plots for each model. For this
artificial data set, the random forest models fit the data perfectly.

``` r
# Obtain predictions for each model
modelPreds <- lapply(myMods, FUN = function(m) {
  data.frame(prob = predict(m[[1]], m[[1]][["trainingData"]], type = "prob")[, c("Walking")]) %>% 
    mutate(estActivity = ifelse(prob >= 0.50, "Walking", "Other"),
           Activity = m[[1]][["trainingData"]][[".outcome"]])
  })

# Plotting confusion matrix plots
lapply(1:3, FUN = function(m) {
    lapply(modelPreds[(1 + (m-1)*3):(3*m)], FUN = function(modelPred){
      mygg <- modelPred %>% confusionPlot() 
      mygg[[2]] %>% ggplot2::ggplotGrob()}) %>% 
        do.call(what = 'cbind')}) %>% do.call(what = 'rbind') %>% grid::grid.draw()
```

### Threshold Algorithm

Let’s also implement the threshold algorithm for detection of walking
for physical activity of Mays et al. (2020).

``` r
# Implementing exercise detection algorithm
algResults <- threshAlg(steps = simData$steps, thresh = "adaptive")

algResults[[1]] %>% mutate(Activity = simData$Activity) %>% confusionPlot()
```

    ## [[1]]
    ## NULL
    ## 
    ## [[2]]

We can also display the estimated and true activities across time using
the `activityPlot()` function.

``` r
# Plotting true and estimated activities
simData %>% mutate(estActivity = algResults[[1]]$estActivity) %>% activityPlot()
```

<h2 id="refs">

References

</h2>

Fiecas, M., Cribben, I., Bahktiari, R., and Cummine, J. (2017). A variance components model for statistical inference on functional connectivity networks. *NeuroImage*, 149, 256-266.

Politis, D.N. and H. White (2004), Automatic block-length selection for the dependent bootstrap, *Econometric Reviews* 23(1), 53-70.

Roy, R. (1989). Asymptotic covariance structure of serial correlations in multivariate time series, *Biometrika*, 76(4), 824-827.


