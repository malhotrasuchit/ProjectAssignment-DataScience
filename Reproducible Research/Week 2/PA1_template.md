Reproducible Research: Peer Assessment 1
================

Loading and preprocessing the data
----------------------------------

We assume that the reader set the correct R working directory with the setwd() function.

### 1. Load the data (i.e. read.csv())

We use the aggregate function, removing NAs, and draw the histogram with base plotting:

``` r
echo = TRUE # The clode is displayed
# Load the raw activity data
activity <- read.csv("activity.csv")
```

What is the mean total number of steps taken per day?
-----------------------------------------------------

For this part of the assignment, you can ignore the missing values in the dataset.

### 1. Total number of steps taken per day

``` r
activity_steps_day <- aggregate(steps ~ date, data = activity, FUN = sum, na.rm = TRUE)
```

### 2. Histogram of the total number of steps taken each day.

``` r
hist(activity_steps_day$steps, xlab = "Steps per Day", main = "Total number of steps taken per day", col = "Orange")
```

![](PA1_template_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-3-1.png)

Note: we could set the number of breaks (bins) to values higher than 5, but we chose not to, the insights seem sufficient at this stage.

### 3. Mean and median of the total number of steps taken per day

``` r
mean_steps <- mean(activity_steps_day$steps)
median_steps <- median(activity_steps_day$steps)
#we set a normal number format to display the results
mean_steps <- format(mean_steps,digits=1)
median_steps <- format(median_steps,digits=1)
```

#### Mean of Total Number of steps taken per day is **10766**

#### Median of Total Number of steps taken per day is **10765**

What is the average daily activity pattern?
-------------------------------------------

### 1. Time series plot

Time series plot of the 5-minute interval (x-axis) and the average number of steps taken, averaged across all days (y-axis)

``` r
#Aggregate function for mean over all days, for each interval
activity_steps_mean <- aggregate(steps ~ interval, data = activity, FUN = mean, na.rm = TRUE)
#Plot
plot(activity_steps_mean$interval, activity_steps_mean$steps, type = "l", col = "tan3", xlab = "Intervals", ylab = "Total steps per interval", main = "Number of steps per interval (averaged) (NA removed)")
```

![](PA1_template_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-5-1.png)

### 2. Which 5-minute interval, on average across all the days in the dataset, contains the maximum number of steps?

``` r
#what is the highest steps value? (maximum of steps on one given interval)
max_steps <-max(activity_steps_mean$steps)
#for which interval are the numbers of steps per interval at the highest?
max_interval <- activity_steps_mean$interval[which(activity_steps_mean$steps == max_steps)]
max_steps <- round(max_steps, digits = 2)
```

Imputing missing values
-----------------------

### 1. Calculate total number of missing values in the dataset

``` r
sum(is.na(activity))
```

    ## [1] 2304

This corresponds to the summary results found at the top of document, all NA values found in steps variable only.

### 2. Devise a strategy for filling in all of the missing values in the dataset

We run a couple of unsophisticated charts to decide which unsophisticated strategy we will adopt

``` r
#subset general dataset with missing values only
missing_values <- subset(activity, is.na(steps))
#plot repartition, by date or by intervals
par(mfrow = c(2,1), mar = c(2, 2, 1, 1))
hist(missing_values$interval, main="NAs repartition per interval")
hist(as.numeric(missing_values$date), main = "NAs repartion per date", breaks = 61)
```

![](PA1_template_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-8-1.png) We see that NAs run equally over all intervals. On the other hand, checking with dates, we see all NAâs are spread between 8 specific days only. To reduce that effect, best will be to take the mean for missing interval across all the days in the dataset.

### 3. Create new dataset with the missing data filled in

We will follow these steps to replace the missing data: - calculate average steps per interval, across all the days - cut the activity dataset in two parts: activity\_NAs, activity\_non\_NAs - in activity\_NAs, we convert, we convert the steps variable into a factor, then replace levels with new computed values - a bit of formatting to convert the steps vector variable into an integer vector - merge/rbind the two datasets into a new dataset, with missing data filled in

``` r
# calculate mean of steps per interval, we end up with a mean for all 288 intervals
MeanStepsPerInterval <- tapply(activity$steps, activity$interval, mean, na.rm = TRUE)
# cut the 'activity' dataset in 2 parts (with and without NAs)
activity_NAs <- activity[is.na(activity$steps),]
activity_non_NAs <- activity[!is.na(activity$steps),]
#replace missing values in activity_NAs
activity_NAs$steps <- as.factor(activity_NAs$interval)
levels(activity_NAs$steps) <- MeanStepsPerInterval
#change the vector back as integer 
levels(activity_NAs$steps) <- round(as.numeric(levels(activity_NAs$steps)))
activity_NAs$steps <- as.integer(as.vector(activity_NAs$steps))
#merge/rbind the two datasets together
imputed_activity <- rbind(activity_NAs, activity_non_NAs)
```

### 4. Make a histogram of the total number of steps taken each day

``` r
#Plotting parameters to place previous histogram and new one next to each other
par(mfrow = c(1,2))
#Plot again the histogram from the first part of the assignment
activity_steps_day <- aggregate(steps ~ date, data = activity, FUN = sum, na.rm = TRUE)
hist(activity_steps_day$steps, xlab = "Steps per Day", main = "NAs REMOVED - Total steps/day", col = "wheat")
#Plot new histogram, with imputed missing values
imp_activity_steps_day <- aggregate(steps ~ date, data = imputed_activity, FUN = sum, na.rm = TRUE)
hist(imp_activity_steps_day$steps, xlab = "Steps per Day", main = "NAs IMPUTED - Total steps/day", col = "wheat")
```

![](PA1_template_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-10-1.png)

We calculate like previously the mean and median values, and store the new and old results in a data frame for easier comparison:

``` r
imp_mean_steps <- mean(imp_activity_steps_day$steps)
imp_median_steps <- median(imp_activity_steps_day$steps)
#we set a normal number format to display the results
imp_mean_steps <- format(imp_mean_steps,digits=1)
imp_median_steps <- format(imp_median_steps,digits=1)
#store the results in a dataframe
results_mean_median <- data.frame(c(mean_steps, median_steps), c(imp_mean_steps, imp_median_steps))
colnames(results_mean_median) <- c("NA removed", "Imputed NA values")
rownames(results_mean_median) <- c("mean", "median")
library(xtable)
```

    ## Warning: package 'xtable' was built under R version 3.4.4

``` r
#We use the xtable package to print the table with all values:
xt <- xtable(results_mean_median)
print(xt, type  = "html")
```

    ## <!-- html table generated in R 3.4.0 by xtable 1.8-2 package -->
    ## <!-- Mon May 14 14:11:17 2018 -->
    ## <table border=1>
    ## <tr> <th>  </th> <th> NA removed </th> <th> Imputed NA values </th>  </tr>
    ##   <tr> <td align="right"> mean </td> <td> 10766 </td> <td> 10766 </td> </tr>
    ##   <tr> <td align="right"> median </td> <td> 10765 </td> <td> 10762 </td> </tr>
    ##    </table>

CONCLUSIONS: Imputing missing values didnât change the mean value whatsoever, the median value is reduced only by 0.027% (3/10765\*100), which is as good as nothing. Both histograms show the same structure, with imputed NAs we notice however higher frequencies.

Are there differences in activity patterns between weekdays and weekends?
-------------------------------------------------------------------------

### 1. Create a new factor variable

``` r
#elseif function to categorize Saturday and Sunday as factor level "weekend", all the rest as "weekday"
imputed_activity$dayType <- ifelse(weekdays(as.Date(imputed_activity$date)) == "Samstag" | weekdays(as.Date(imputed_activity$date)) == "Sonntag", "weekend", "weekday")
#transform dayType variable into factor
imputed_activity$dayType <- factor(imputed_activity$dayType)
```

### 2. Panel plot containing time series plot

``` r
#Aggregate a table showing mean steps for all intervals, acrlss week days and weekend days
steps_interval_dayType <- aggregate(steps ~ interval + dayType, data = imputed_activity, FUN = mean)
#verify new dataframe 
head(steps_interval_dayType)
```

    ##   interval dayType      steps
    ## 1        0 weekday 1.75409836
    ## 2        5 weekday 0.29508197
    ## 3       10 weekday 0.11475410
    ## 4       15 weekday 0.13114754
    ## 5       20 weekday 0.06557377
    ## 6       25 weekday 2.08196721

``` r
#add descriptive variables
names(steps_interval_dayType) <- c("interval", "day_type", "mean_steps")
#plot with ggplot2
library(ggplot2)
```

    ## Warning: package 'ggplot2' was built under R version 3.4.1

``` r
plot <- ggplot(steps_interval_dayType, aes(interval, mean_steps))
plot + geom_line(color = "tan3") + facet_grid(day_type~.) + labs(x = "Intervals", y = "Average Steps", title = "Activity Patterns")
```

![](PA1_template_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-14-1.png)

CONCLUSION: it seems the tested subjects have an earlier start in the week days with a peak between 8am and 9am. On weekends, the activity seems more spread between 8am and 8pm.
