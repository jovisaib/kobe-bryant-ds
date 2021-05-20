
library(caret)
library(data.table)
library(corrplot)
library(Matrix)


data <- as.data.frame(fread("data.csv", header = T, stringsAsFactors = T))

summary(data)
str(data)
table(is.na(data))


ggplot(data, aes(shot_type, fill=shot_type)) + geom_bar()
ggplot(data, aes(shot_zone_area, fill=shot_zone_area)) + geom_bar()
ggplot(data, aes(shot_zone_basic, fill=shot_zone_basic)) + geom_bar()
ggplot(data, aes(shot_zone_range, fill=shot_zone_range)) + geom_bar()
ggplot(data, aes(combined_shot_type, fill=combined_shot_type)) + geom_bar()
ggplot(data, aes(shot_made_flag, fill=shot_made_flag)) + geom_bar()

#Discretizar
#ggplot(data, aes(matchup)) + geom_bar()
#ggplot(data, aes(opponent)) + geom_bar()
#ggplot(data, aes(season)) + geom_bar() 


ggplot(data, aes(x = loc_x, y = loc_y)) +
  geom_point(data[data$combined_shot_type=="Jump Shot",], mapping=aes(x = loc_x, y = loc_y),colour = "grey", alpha=0.3) +
  geom_point(data[data$combined_shot_type!="Jump Shot",], mapping=aes(x = loc_x, y = loc_y, colour = combined_shot_type), alpha=0.6) +
  labs(title="Shot type") + theme_void()

ggplot(data, aes(x = loc_x, y = loc_y)) +
  geom_point(data, mapping=aes(x = loc_x, y = loc_y, colour=shot_zone_basic), alpha=0.3) +
  labs(title="Shot zones") + theme_void()

ggplot(data, aes(x = loc_x, y = loc_y)) +
  geom_point(data, mapping=aes(x = loc_x, y = loc_y, colour=shot_zone_range), alpha=0.3) +
  labs(title="Shot zone range") + theme_void()


ggplot(data, aes(x = loc_x, y = loc_y)) +
  geom_point(data, mapping=aes(x = loc_x, y = loc_y, colour=shot_zone_area), alpha=0.3) +
  labs(title="Shot zone area") + theme_void()



numerical <- subset(data, select=c(playoffs, minutes_remaining, seconds_remaining,shot_distance, lat, loc_y, loc_x, lon))
#pairs(numerical)
correlations <- cor(numerical)
correlations
corrplot(correlations)
palette = colorRampPalette(c("green", "white", "red")) (20)
heatmap(x = correlations, col = palette, symm = TRUE)


pureData <- subset(data, select = -c(action_type, seconds_remaining, minutes_remaining, team_id, team_name, game_event_id, game_id, lat, lon, game_date, matchup))
pureData$seconds_from_period_end = 60 * data$minutes_remaining + data$seconds_remaining
pureData$localmatch <- ifelse(grepl("vs", data$matchup), 1, 0)
pureData$shot_type <- as.integer(data$shot_type)
pureData$last5secs <- (pureData$seconds_from_period_end < 5) * 1


categorical = subset(data, select=c(period, season, combined_shot_type, shot_zone_area, shot_zone_basic, shot_zone_range, opponent))
dummy <- dummyVars(" ~ .", data = categorical)
ohdata <- as.data.frame(predict(dummy, newdata = categorical))
names(ohdata) <- make.names(colnames(ohdata))

pureData <- subset(pureData, select=-c(period, season, combined_shot_type, shot_zone_area, shot_zone_basic, shot_zone_range, opponent))
pureData <- cbind(pureData[,!names(pureData) %in% categorical], ohdata)

str(pureData)
train <- subset(pureData, !is.na(pureData$shot_made_flag))
test <- subset(pureData, is.na(pureData$shot_made_flag))
testids <- test$shot_id
train$shot_id <- NULL
test$shot_id <- NULL

library(MASS)
set.seed(200)
dfl <- lda(shot_made_flag ~ ., train)
pred.dfl <- predict(dfl, train)
trainig_error <- mean(train$shot_made_flag != pred.dfl$class) * 100
paste("Trainig_error =", trainig_error, "%")
confusionMatrix(as.factor(pred.dfl$class), as.factor(train$shot_made_flag))




LogLoss<-function(actual, predicted)
{
  predicted<-(pmax(predicted, 0.00001))
  predicted<-(pmin(predicted, 0.99999))
  result<- (-1/length(actual))*(sum((actual*log(predicted)+(1-actual)*log(1- predicted))))
  return(result)
}
#LogLoss(train$shot_made_flag, as.integer(pred.dfl$class))


library(xgboost)
xtrain <- sparse.model.matrix(shot_made_flag ~ .-1, data=train)
dtrain <- xgb.DMatrix(data=xtrain, label=train$shot_made_flag)
set.seed(200)
xgb <- xgb.train(
  data = dtrain,
  max.depth = 2,
  eta = 1,
  nthread = 2,
  nrounds = 2,
  eval_metric = "logloss",
  objective = "binary:logistic",
  verbose = 1
)
predict_xgb <- predict(xgb, xtrain)
LogLoss(as.integer(train$shot_made_flag), as.integer(predict_xgb))

train$shot_made_flag <- as.factor(train$shot_made_flag)

library(randomForest)
set.seed(200)
rf = randomForest(shot_made_flag~., data=train, ntree=100, importance=T)
rf
prediction_for_table <- predict(rf, train)
trainig_error <- mean(train$shot_made_flag != prediction_for_table) * 100
paste("Trainig_error =", trainig_error, "%")
LogLoss(as.integer(train$shot_made_flag), as.integer(prediction_for_table))
confusionMatrix(prediction_for_table, as.factor(train$shot_made_flag))
missclassified <- sum(train$shot_made_flag != prediction_for_table)
paste("Total missclasified =", missclassified)



library(rpart)
set.seed(200)
fit <- rpart(shot_made_flag~., data = train, method = 'class')
p <- predict(fit, train, type = 'class')
trainig_error <- mean(train$shot_made_flag != p) * 100
paste("Trainig_error =", trainig_error, "%")
confusionMatrix(p, as.factor(train$shot_made_flag))
missclassified <- sum(train$shot_made_flag != p)
paste("Total missclasified =", missclassified)



test$shot_made_flag <- -1
xtest <- sparse.model.matrix(shot_made_flag ~.-1, data=test)
predxgb <- predict(xgb, xtest)
samples <- data.frame(shot_id = testids, shot_made_flag = predxgb)
write.csv(samples, file = "sumbission.csv", row.names = F)


