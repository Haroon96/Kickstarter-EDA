Data <- read.csv(file = "data.csv") 

Data <- subset.data.frame(Data, state != "canceled" & state != "live" & state != "suspended" & state != "undefined"  & currency == "USD")


# Change the point size, and shape
ggplot(Data, aes(x=goal, y=state)) +
geom_point(size=2, shape=19)

#bar chart stuff

#creat a table which contains frequencies of main_category column
cat.freq <- table(Data$main_category)

barplot(cat.freq)

#show bar chat in descending order
barplot(cat.freq[order(cat.freq, decreasing = T)])

#for the count and other aggregation function
library(plyr)
require(speedglm)
library(CRAN)

#count the frequency of each state label for each main_category
library(plyr)
count(Data, c("main_category", "state"))
max(Data, c("goal"))

#get difference b/w deadline and launch in days
duration<- difftime(Data$deadline ,Data$launched , units = c("days"))

#drop currency and country columns cause only 1 value messes up the model + mem constraints
df <- subset(Data, select = -c(category, main_category, currency, deadline, launched, country, name, pledged,  backers, usd_pledged_real, usd_goal_real))

#remove all columns with N/A from the df
df[complete.cases(df),]

#add duration column to dataframe
df$duration<- duration

#add Goal per day to df

GPD <- df$goal / as.numeric(df$ duration)
df$GPD<- GPD

traindf <- df[1:200000, ]
testdf <- df[200001:261511,]

#run logistic regression
fit <- glm(state~., data=traindf, family=binomial)


#summary and prediction stuff
summary(fit)
prediction <- predict(fit, type="response", newdata=testdf)

fittedresults <- ifelse(prediction > 0.5,"successful","failed")

fittedresults<-as.factor(fittedresults)

#confusion matrix
require(caret)   
library(caret)

cm<-confusionMatrix(data=fittedresults, 
                    reference=testdf$state)

#show results
cm












