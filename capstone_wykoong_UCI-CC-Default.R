


######################### IMPORTANT #########################
#
# THIS R SCRIP INCLUDE APPENDIX A & APPENDIX B FOR REFERENCE PURPOSES.
#  - APPENDIX A MAY TAKE APPROXIMATE 1 HOUR FOR COMPLETE EXECUTION, AND
#  - APPENDIX B MAY TAKE MORE THAN 12 HOURS FOR COMPLETE EXECUTION. 
# RESULT OF THESE APPENDIXES ARE INCLUDED IN MAIN PORTION. 
# COPIES OF RESULT ALSO INCLUDED IN README FILE.
# 
# RUN ONLY IF YOU COMFORTABLE WITH THE LENGTH EXECUTION TIME, 
# OR YOU HAVE A POWERFUL MACHINE.
#
######################### IMPORTANT #########################


# title: 'UCI Credit Card Default'
# subtitle: 'HarvardX (PH125.9x) - Data Science: Capstone'
# author: "Wayne Koong Wah Yan"
# email: "wykoong@gmail.com"
# date: "10/26/2020"

############################# . #############################
#
#    LIBRARY & ENVIRONMENT VARIABLES
#    
############################# . #############################


library(tidyverse)
library(caret)
library(data.table)
library(formattable)
library(corrplot)
library(randomForest)
library(stringr)
library(xgboost)

############################# . #############################
#    OBTAIN DATA
#    Data is downloaded from Kaggle.com, following is the URL:
#      https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset
############################# . #############################

dat_raw <- read.csv("https://raw.githubusercontent.com/wykoong/PH125.9x-Capstone-UCI_CreditCard_DEFAULT/master/data/UCI_Credit_Card.csv")
dat_raw_original <- dat_raw         # Keep a copy of original data for comparison

#---------------------------- . ----------------------------#
#    Analyzed downloaded data set
#---------------------------- . ----------------------------#
summary(dat_raw, title = "CREDIT CARD DEFAULT")

names(dat_raw)
# [1] "ID"                         "LIMIT_BAL"                  "SEX"                       
# [4] "EDUCATION"                  "MARRIAGE"                   "AGE"                       
# [7] "PAY_0"                      "PAY_2"                      "PAY_3"                     
# [10] "PAY_4"                      "PAY_5"                      "PAY_6"                     
# [13] "BILL_AMT1"                  "BILL_AMT2"                  "BILL_AMT3"                 
# [16] "BILL_AMT4"                  "BILL_AMT5"                  "BILL_AMT6"                 
# [19] "PAY_AMT1"                   "PAY_AMT2"                   "PAY_AMT3"                  
# [22] "PAY_AMT4"                   "PAY_AMT5"                   "PAY_AMT6"                  
# [25] "default.payment.next.month"

#---------------------------- . ----------------------------#
#    COLUMN DEFINITION AS KAGGLE.COM
#      
#---------------------------- . ----------------------------#
# ID: ID of each client
# LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
# SEX: Gender (1=male, 2=female)
# EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# MARRIAGE: Marital status (1=married, 2=single, 3=others)
# AGE: Age in years
# PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
# PAY_2: Repayment status in August, 2005 (scale same as above)
# PAY_3: Repayment status in July, 2005 (scale same as above)
# PAY_4: Repayment status in June, 2005 (scale same as above)
# PAY_5: Repayment status in May, 2005 (scale same as above)
# PAY_6: Repayment status in April, 2005 (scale same as above)
# BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
# BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
# BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
# BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
# BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
# BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
# PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
# PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
# PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
# PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
# PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
# PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
# default.payment.next.month: Default payment (1=yes, 0=no)


#---------------------------- . ----------------------------#
#    Removed unwanted ID column
#      ID column is the unique identifier of the data and play no role in 
#        analysis but mislead models if accidentally included.
#---------------------------- . ----------------------------#
dat_raw <- dat_raw[,-1]

#---------------------------- . ----------------------------#
#    Simplified column name default.payment.next.month to DEFAULT 
#      Column name default.payment.next.month is kind of too long & not in
#        consistent format as other column.
#---------------------------- . ----------------------------#
#
# What does default.payment.next.month means here, 
# According to Wikipedia: 
#   To default is to fail to make a payment on a debt by the due date. 
#   If this happens with a credit card, creditors might raise interest rates to the default (or penalty rate) or decrease the line of credit. In case of serious delinquency, the card issuer can even take legal action to enforce payment or to garnish wages

names(dat_raw)[24] <- "DEFAULT"

#---------------------------- . ----------------------------#
#    Rename column name PAY_0 to PAY_1 
#      Column is renamed to made in consistent with BILL_AMT1 & PAY_AMT1
#        which mean that this is Sep 2005 data.
#---------------------------- . ----------------------------#
names(dat_raw)[6] <- "PAY_1"

#---------------------------- . ----------------------------#
#    Rename column name FOR EASY REFERENCE THROUGHOUT THE REPORT 
#      - BILL_AMTx to BILLED_x
#      - PAY_AMTx to PAYMENT_x
#---------------------------- . ----------------------------#
colnames(dat_raw)[12:17] <- rep(paste("BILLED",1:6,sep="_"),1)
colnames(dat_raw)[18:23] <- rep(paste("PAYMENT",1:6,sep="_"),1)

names(dat_raw)
dim(dat_raw)

matrix(c(VARIABLE=c("ID", "LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",
                    "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
                    "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5",'BILL_AMT6',
                    "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6",
                    "default.payment.next.month"),
         DESCRIPTION <- c("ID of each client", 
                          "Credit Limit, Amount of given credit in NT dollars (includes individual and family/supplementary credit", 
                          "Gender (1=male, 2=female)",
                          "Education (0=?, 1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)", 
                          "Marital status (0=?,1=married, 2=single, 3=others)", 
                          "Age in years",
                          "Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)",
                          "Repayment status in August, 2005 (scale same as above)",
                          "Repayment status in July, 2005 (scale same as above)",
                          "Repayment status in June, 2005 (scale same as above)",
                          "Repayment status in May, 2005 (scale same as above)",
                          "Repayment status in April, 2005 (scale same as above)",
                          "Amount of bill statement in September, 2005 (NT dollar)",
                          "Amount of bill statement in August, 2005 (NT dollar)",
                          "Amount of bill statement in July, 2005 (NT dollar)",
                          "Amount of bill statement in June, 2005 (NT dollar)",
                          "Amount of bill statement in May, 2005 (NT dollar)",
                          "Amount of bill statement in April, 2005 (NT dollar)",
                          "Amount of previous payment in September, 2005 (NT dollar)",
                          "Amount of previous payment in August, 2005 (NT dollar)",
                          "Amount of previous payment in July, 2005 (NT dollar)",
                          "Amount of previous payment in June, 2005 (NT dollar)",
                          "Amount of previous payment in May, 2005 (NT dollar)",
                          "Amount of previous payment in April, 2005 (NT dollar)",
                          "Default payment (1=yes, 0=no)")),
       25,2)  %>% knitr::kable(caption="PREDICTION RESULT", digits=4)



############################# . #############################
#
#    SCRUB DATA
#    
############################# . #############################

#---------------------------- . ----------------------------#
#    NA ANALYSIS  
#---------------------------- . ----------------------------#
anyNA(dat_raw)
summary(dat_raw)

#---------------------------- . ----------------------------#
#    CATEGORIZED DATA
#---------------------------- . ----------------------------#
tmp_categorical <- c('SEX','EDUCATION','MARRIAGE','DEFAULT')
dat_raw[tmp_categorical] <- lapply(dat_raw[tmp_categorical], 
                                   function(x) as.factor(x))
rm(tmp_categorical)

#---------------------------- . ----------------------------#
#    EXAMINE SEX COLUMN
#    - SEX: Gender (1=male, 2=female)
#---------------------------- . ----------------------------#
unique(dat_raw$SEX)
# NO PARSING REQUIRED

dat_raw %>% group_by(SEX) %>% summarise(Count=n()) %>% 
  mutate(SEX=ifelse(SEX==1,"MALE","FEMALE"), CountLabel = comma(Count,digits=0)) %>%
  ggplot(aes(x="", y=Count, fill=SEX)) +
  geom_bar(width = 1, stat = "identity") + 
  coord_polar("y", start=0) + xlab("") + ylab("") + ggtitle("SEX Distributionn") +
  geom_text(aes(y=Count, label=SEX), color="white", vjust=c(-7,-1), size=5) +
  geom_text(aes(y=Count, label=CountLabel), color="white", vjust=c(-7,0), size=4) +
  scale_fill_manual(values=c("blue", "red")) + 
  theme_minimal() + theme(legend.position="none") + 
  theme(plot.title = element_text(hjust = 0.5))


#---------------------------- . ----------------------------#
#    PARSE EDUCATION COLUMN
#      - EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
#      - 0,4,5 & 6 are all unknown data, reset to 4
#---------------------------- . ----------------------------#
dat_raw$EDUCATION[which(dat_raw$EDUCATION %in% c(0,4,5,6))] = 4

dat_raw_original %>% group_by(EDUCATION) %>% summarise(Count=n()) %>% 
  mutate(Edu_text = case_when(
    EDUCATION == 1 ~ "Graduate School",
    EDUCATION == 2 ~ "University",
    EDUCATION == 3 ~ "High School",
    TRUE ~ "Others"
  ), Edu_Label = paste(Edu_text, "(", comma(Count,digits=0),")")) %>%
  ggplot(aes(x="", y=Count, fill=Edu_Label)) +
  geom_bar(width = 1, stat = "identity") + 
  coord_polar("y", start=0) + xlab("") + ylab("") + ggtitle("EDUCATION DISTRIBUTION (BEFORE PARSE)") +
  scale_fill_manual(values=c("blue", "red", "green", "yellow", "pink", "purple", "orange")) + 
  theme_minimal() + labs(fill = "Education")+
  theme(plot.title = element_text(hjust = 0.5), legend.justification = "center",
        legend.position = "top")

dat_raw %>% group_by(EDUCATION) %>% summarise(Count=n()) %>% 
  mutate(Edu_text = case_when(
    EDUCATION == 1 ~ "Graduate School",
    EDUCATION == 2 ~ "University",
    EDUCATION == 3 ~ "High School",
    TRUE ~ "Others"
  ), Edu_Label = paste(Edu_text, "(", comma(Count,digits=0),")")) %>%
  ggplot(aes(x="", y=Count, fill=Edu_Label)) +
  geom_bar(width = 1, stat = "identity") + 
  coord_polar("y", start=0) + xlab("") + ylab("") + ggtitle("EDUCATION DISTRIBUTION (AFTER PARSE)") +
  scale_fill_manual(values=c("blue", "red", "green", "yellow", "pink", "purple", "orange")) + 
  theme_minimal() + labs(fill = "Education")+
  theme(plot.title = element_text(hjust = 0.5), legend.justification = "center",
        legend.position = "top")

#---------------------------- . ----------------------------#
#    PARSE MARRIAGE COLUMN
#      - MARRIAGE: Marital status (1=married, 2=single, 3=others)
#      - 0 & 3 are both others, reset to 3
#---------------------------- . ----------------------------#
dat_raw$MARRIAGE[which(dat_raw$MARRIAGE %in% c(0,3))] = 3

dat_raw_original %>% group_by(MARRIAGE) %>% summarise(Count=n()) %>% 
  mutate(Mar_text = case_when(
    MARRIAGE == 1 ~ "Married",
    MARRIAGE == 2 ~ "Single",
    TRUE ~ "Others"
  ), Mar_Label = paste(Mar_text, "(", comma(Count,digits=0),")")) %>%
  ggplot(aes(x="", y=Count, fill=Mar_Label)) +
  geom_bar(width = 1, stat = "identity") + 
  coord_polar("y", start=0) + xlab("") + ylab("") + ggtitle("MARRIAGE DISTRIBUTION (BEFORE PARSE)") +
  scale_fill_manual(values=c("blue", "red", "green", "yellow", "pink", "purple", "orange")) + 
  theme_minimal() + labs(fill = "MARRIAGE")+
  theme(plot.title = element_text(hjust = 0.5), legend.justification = "center",
        legend.position = "right")

dat_raw %>% group_by(MARRIAGE) %>% summarise(Count=n()) %>% 
  mutate(Mar_text = case_when(
    MARRIAGE == 1 ~ "Married",
    MARRIAGE == 2 ~ "Single",
    TRUE ~ "Others"
  ), Mar_Label = paste(Mar_text, "(", comma(Count,digits=0),")")) %>%
  ggplot(aes(x="", y=Count, fill=Mar_Label)) +
  geom_bar(width = 1, stat = "identity") + 
  coord_polar("y", start=0) + xlab("") + ylab("") + ggtitle("MARRIAGE DISTRIBUTION (AFTER PARSE)") +
  scale_fill_manual(values=c("blue", "red", "green", "yellow", "pink", "purple", "orange")) + 
  theme_minimal() + labs(fill = "MARRIAGE")+
  theme(plot.title = element_text(hjust = 0.5), legend.justification = "center",
        legend.position = "right")

#---------------------------- . ----------------------------#
#    FACTORIZED AGE using AVERAGE SILHOUETTE
# Silhouette analysis can be used to study the separation distance between the resulting clusters. The silhouette plot displays a measure of how close each point in one cluster is to points in the neighboring clusters and thus provides a way to assess parameters like number of clusters visually. This measure has a range of [-1, 1].
# Silhouette coefficients (as these values are referred to as) near +1 indicate that the sample is far away from the neighboring clusters. A value of 0 indicates that the sample is on or very close to the decision boundary between two neighboring clusters and negative values indicate that those samples might have been assigned to the wrong cluster.
#---------------------------- . ----------------------------#
# tmp_cluster <- numeric(20)
# tmp_k <- seq(2,20)
# tmp_k <- c(2,20)
# 
# tmp_silavg <- sapply(tmp_k, function(x){
#   tmp_start_timer <- Sys.time()
#   tmp_avg <- pam(dat_raw$AGE, x)$silinfo$avg.width
#   cat("silhouette analysis for cluster:", x, 
#       ", average:",tmp_avg,"\n")
#   cat("Now:", print(Sys.time(),"%d.%m.%Y %H:%M:%S"),", proc time:",
#       print(difftime(tmp_start_timer,Sys.time())), "\n")
#   tmp_avg
# })

# silhouette analysis for cluster: 2 , average: 0.60518, proc time: -2.8947 
# silhouette analysis for cluster: 3 , average: 0.56497, proc time: -3.5551 
# silhouette analysis for cluster: 4 , average: 0.54865, proc time: -4.1282 
# silhouette analysis for cluster: 5 , average: 0.539, proc time: -11.115 
# silhouette analysis for cluster: 6 , average: 0.54956, proc time: -5.3508 
# silhouette analysis for cluster: 7 , average: 0.54953, proc time: -3.9899 
# silhouette analysis for cluster: 8 , average: 0.55475, proc time: -10.57 
# silhouette analysis for cluster: 9 , average: 0.5566, proc time: -14.516 
# silhouette analysis for cluster: 10 , average: 0.5774, proc time: -12.304 
# silhouette analysis for cluster: 11 , average: 0.58894, proc time: -10.398 
# silhouette analysis for cluster: 12 , average: 0.58285, proc time: -10.538 
# silhouette analysis for cluster: 13 , average: 0.6149, proc time: -15.127 
# silhouette analysis for cluster: 14 , average: 0.6324, proc time: -16.197 
# silhouette analysis for cluster: 15 , average: 0.66712, proc time: -13.272 
# silhouette analysis for cluster: 16 , average: 0.66235, proc time: -13.642 
# silhouette analysis for cluster: 17 , average: 0.68295, proc time: -13.799 
# silhouette analysis for cluster: 18 , average: 0.68217, proc time: -13.841 
# silhouette analysis for cluster: 19 , average: 0.72059, proc time: -14.705 
# silhouette analysis for cluster: 20 , average: 0.75041, proc time: -11.728 

tmp_silavg <- rbind(cluster=2:20, average=c(0.60518,0.56497,0.54865,0.539,0.54956,0.54953,0.55475,
                                            0.5566,0.5774,0.58894,0.58285,0.6149,0.6324,0.66712,
                                            0.66235,0.68295,0.68217,0.72059,0.75041)) %>% 
  t() %>% as.data.frame() 

tmp_AGE_cluster <- tmp_silavg %>% arrange(desc(average)) %>% top_n(1) %>% select(cluster)

tmp_silavg %>% ggplot(aes(cluster,average)) + geom_line(stat="identity") + 
  geom_text(aes(label=average)) +
  ggtitle("AGE Clustering Assessment") + 
  xlab("Clusters Size") + ylab("Average Silhouette Width") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . #
#    AGE - GROUPING
#      - Based on calculated Average Silhouette - to cluster into 20 clusters
#      - Means each age group contains 3 years:
#      - - 21, 24, 27 ....81
# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . #

dat_raw <- dat_raw %>% mutate(AGE_FAC = round(AGE/3,0)*3)
summary(dat_raw$AGE_FAC)

dat_raw %>%  
  ggplot(aes(AGE_FAC)) + geom_bar() + ggtitle("AGE (FACTORIZED) DISTRUBUTION") +
  xlab("AGE") + ylab("Count") +   theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5))

#---------------------------- . ----------------------------#
#    NUMERIC DATA - MAKING SENSE of BILLED, PAYMENT & LIMIT_BAL
#      - Client can paid partial, full or more than billed amount. 
#      - BILLED amount can be negative, in this scenario, there's debit balance.
#      - PAYMENT amount must be positive. 
#      - - For negative scenario, the amount is set to 0 (Clean PAYMENT to zero for any negative value.)
#      - LIMIT_BAL
#      - - According to SUMMARY, the value is in a good range.
#---------------------------- . ----------------------------#

dat_raw <- dat_raw %>% mutate(PAYMENT_1=ifelse(PAYMENT_1<0,0,PAYMENT_1),
                              PAYMENT_2=ifelse(PAYMENT_2<0,0,PAYMENT_2),
                              PAYMENT_3=ifelse(PAYMENT_3<0,0,PAYMENT_3),
                              PAYMENT_4=ifelse(PAYMENT_4<0,0,PAYMENT_4),
                              PAYMENT_5=ifelse(PAYMENT_5<0,0,PAYMENT_5),
                              PAYMENT_6=ifelse(PAYMENT_6<0,0,PAYMENT_6))


summary(dat_raw$LIMIT_BAL)


#---------------------------- . ----------------------------#
#    LIMIT_BAL
#    - Due to abnormal data, this field is drop from analysis
#      - Client cannot spend more than LIMIT_BAL. 
#      - - However, BILLED_1 can more than LIMIT_BAL after interest charge. 
#      - - But, from data set, some data are not making any sense, as the different between are big %.

#---------------------------- . ----------------------------#
tmp_dat_limitbal_1 <- dat_raw %>% filter(BILLED_1>LIMIT_BAL) %>% 
  mutate(DIFF_PERCENTAGE=(BILLED_1/LIMIT_BAL)-1) %>%
  filter(DIFF_PERCENTAGE>0.2) %>% 
  select(LIMIT_BAL,DIFF_PERCENTAGE,BILLED_1,BILLED_2,PAYMENT_1,PAYMENT_2) %>%
  top_n(5,wt=DIFF_PERCENTAGE)
tmp_dat_limitbal_2 <- dat_raw %>% filter(BILLED_1>LIMIT_BAL) %>% 
  mutate(DIFF_PERCENTAGE=(BILLED_1/LIMIT_BAL)-1) %>%
  filter(DIFF_PERCENTAGE>0.2) %>% filter(BILLED_2==0) %>%
  select(LIMIT_BAL,DIFF_PERCENTAGE,BILLED_1,BILLED_2,PAYMENT_1,PAYMENT_2) %>%
  top_n(3,wt=DIFF_PERCENTAGE)
tmp_dat_limitbal_3 <- dat_raw %>% filter(BILLED_1>LIMIT_BAL) %>% 
  mutate(DIFF_PERCENTAGE=(BILLED_1/LIMIT_BAL)-1) %>%
  filter(DIFF_PERCENTAGE>0.2) %>% filter(BILLED_2>0 & PAYMENT_2<=0) %>%
  select(LIMIT_BAL,DIFF_PERCENTAGE,BILLED_1,BILLED_2,PAYMENT_1,PAYMENT_2) %>%
  top_n(3,wt=DIFF_PERCENTAGE)
rbind(tmp_dat_limitbal_1,tmp_dat_limitbal_2,tmp_dat_limitbal_3) %>% mutate(LIMIT_BAL=comma(LIMIT_BAL,digits=0)) %>%
  mutate(BILLED_1=comma(BILLED_1,digits=0)) %>%mutate(BILLED_2=comma(BILLED_2,digits=0)) %>%
  mutate(PAYMENT_1=comma(PAYMENT_1,digits=0)) %>%mutate(PAYMENT_2=comma(PAYMENT_2,digits=0)) %>%
  knitr::kable(caption="ABNORMAL DATA", digits=4)
rm(tmp_dat_limitbal_1,tmp_dat_limitbal_2,tmp_dat_limitbal_3)

#---------------------------- . ----------------------------#
#    PARSE PAY_x COLUMN
#      - According to provide column definition
#      - - VALUE = -1 means pay duly
#      - - VALUE = 1 to 8 mean number of months payment delayed
#      - - VALUE = 9 means payment delayed for nine months or more
#      - - Thus valid value should be -1,1..9
#      - To ensure consistent, redefine 0 means pay on time.
#      - Paste value -2 & -1 to 0
#---------------------------- . ----------------------------#

dat_raw %>% filter(PAY_1>=8) %>% top_n(10) %>% 
  select(PAY_1,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6) %>%
  mutate(PAID_1=format(as.Date("1SEP2005","%d%B%Y")+months(PAY_1),"%b %Y"),
         PAID_2=format(as.Date("1AUG2005","%d%B%Y")+months(PAY_2),"%b %Y"),
         PAID_3=format(as.Date("1JUL2005","%d%B%Y")+months(PAY_3),"%b %Y"),
         PAID_4=format(as.Date("1JUN2005","%d%B%Y")+months(PAY_4),"%b %Y"),
         PAID_5=format(as.Date("1MAY2005","%d%B%Y")+months(PAY_5),"%b %Y"),
         PAID_6=format(as.Date("1APR2005","%d%B%Y")+months(PAY_6),"%b %Y"))


# table(c(dat_raw$PAY_1,dat_raw$PAY_2,dat_raw$PAY_3,dat_raw$PAY_4,dat_raw$PAY_5,dat_raw$PAY_6))
# dat_raw$PAY_1[which(dat_raw$PAY_1<0)] = 0
# dat_raw$PAY_2[which(dat_raw$PAY_2<0)] = 0
# dat_raw$PAY_3[which(dat_raw$PAY_3<0)] = 0
# dat_raw$PAY_4[which(dat_raw$PAY_4<0)] = 0
# dat_raw$PAY_5[which(dat_raw$PAY_5<0)] = 0
# dat_raw$PAY_6[which(dat_raw$PAY_6<0)] = 0
# table(c(dat_raw$PAY_1,dat_raw$PAY_2,dat_raw$PAY_3,dat_raw$PAY_4,dat_raw$PAY_5,dat_raw$PAY_6))


#---------------------------- . ----------------------------#
#    ADDING TEXT LABEL TO CATEGORICAL DATA
#---------------------------- . ----------------------------#
dat_raw <- dat_raw %>% mutate(EDU_text = case_when(
  EDUCATION == 1 ~ "Graduate School",
  EDUCATION == 2 ~ "University",
  EDUCATION == 3 ~ "High School",
  TRUE ~ "Others"
)) %>% mutate(SEX_text=ifelse(SEX==1,"MALE","FEMALE")) %>%
  mutate(MAR_text = case_when(
    MARRIAGE == 1 ~ "Married",
    MARRIAGE == 2 ~ "Single",
    TRUE ~ "Others"
  )) %>% mutate(DEF_text=ifelse(DEFAULT==1,"YES","NO"))

names(dat_raw)


############################# . #############################
#
#    DATA EXPLORATION
#                 
############################# . #############################

#---------------------------- . ----------------------------#
#    DEFAULT
#    - OVERALL CHART
#---------------------------- . ----------------------------#
names(dat_raw)
dat_raw %>% select(DEF_text,SEX_text, EDU_text,MAR_text) %>% gather(CATEGORY,VALUE,-DEF_text) %>%
  mutate(CATEGORY = case_when(
    CATEGORY == "SEX_text" ~ "SEX",
    CATEGORY == "EDU_text" ~ "EDUCATION",
    CATEGORY == "MAR_text" ~ "MARRIAGE"
  )) %>%
  ggplot(aes(DEF_text,fill=VALUE)) + geom_bar() + 
  ggtitle("DEFAULT DISTRIBUTION BY EDUCATION, MARRIAGE & SEX") +
  theme_minimal() + theme(plot.title = element_text(hjust = 0.5)) +
  xlab("DEFAULT") +
  facet_grid(~CATEGORY) 

#---------------------------- . ----------------------------#
#    EDUCATION vs DEFAULT
#---------------------------- . ----------------------------#

dat_raw %>%  ggplot(aes(EDU_text,fill=DEF_text)) + geom_bar() +
  ggtitle("COUNT OF DEFAULT BY EDUCATION") + xlab("EDUCATION") + ylab("Count") +
  theme_minimal() + theme(plot.title = element_text(hjust = 0.5)) + 
  scale_fill_manual(name="DEFAULT", values=c("blue", "red"))

tmp_edu <- dat_raw %>% group_by(EDU_text,DEFAULT) %>% summarise(Count=n()) %>% 
  spread(EDU_text,Count) 
tmp_edu <- apply(tmp_edu,2,function(x){as.numeric(x)*100/sum(as.numeric(x))})  
tmp_edu %>% as.data.frame() %>% gather("EDUCATION","VALUE",-DEFAULT) %>% 
  mutate(DEFAULT=ifelse(DEFAULT==100,"YES","NO")) %>%
  mutate(per=comma(VALUE,digits=2)) %>%
  mutate(per=paste(per,"%")) %>%
  ggplot(aes(EDUCATION,VALUE, fill=DEFAULT)) + 
  geom_bar(position = "stack", stat = "identity") +
  geom_text(aes(y=VALUE, label=per), color="white", size=3, vjust=1) +
  theme_minimal() +
  scale_fill_manual(values=c("blue", "red")) +
  ggtitle("PERCENTAGE(%) OF DEFAULT BY EDUCATIONS") + ylab("PERCENTAGE(%)") +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "top")

#---------------------------- . ----------------------------#
#    SEX vs DEFAULT
#---------------------------- . ----------------------------#

dat_raw %>%  ggplot(aes(SEX_text,fill=DEF_text)) + geom_bar() +
  ggtitle("COUNT OF DEFAULT BY SEX") + xlab("SEX") + ylab("Count") +
  theme_minimal() + theme(plot.title = element_text(hjust = 0.5)) + 
  scale_fill_manual(name="DEFAULT", values=c("blue", "red"))

tmp_sex <- dat_raw %>% group_by(SEX_text,DEFAULT) %>% summarise(Count=n()) %>% 
  spread(SEX_text,Count) 
tmp_sex <- apply(tmp_sex,2,function(x){as.numeric(x)*100/sum(as.numeric(x))})  
tmp_sex %>% as.data.frame() %>% gather("SEX","VALUE",-DEFAULT) %>% 
  mutate(DEFAULT=ifelse(DEFAULT==100,"YES","NO")) %>%
  mutate(per=comma(VALUE,digits=2)) %>%
  mutate(per=paste(per,"%")) %>%
  ggplot(aes(SEX,VALUE, fill=DEFAULT)) + 
  geom_bar(position = "stack", stat = "identity") +
  geom_text(aes(y=VALUE, label=per), color="white", size=5, vjust=1) +
  theme_minimal() +
  scale_fill_manual(values=c("blue", "red")) +
  ggtitle("PERCENTAGE(%) OF DEFAULT BY SEX") + ylab("PERCENTAGE(%)") +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "top")


#---------------------------- . ----------------------------#
#    MARRIAGE vs DEFAULT
#---------------------------- . ----------------------------#
dat_raw %>%  ggplot(aes(MAR_text,fill=DEF_text)) + geom_bar() +
  ggtitle("COUNT OF DEFAULT BY MARRIAGE") + xlab("MARRIAGE") + ylab("Count") +
  theme_minimal() + theme(plot.title = element_text(hjust = 0.5)) + 
  scale_fill_manual(name="DEFAULT", values=c("blue", "red"))

tmp_mar <- dat_raw %>% group_by(MAR_text,DEFAULT) %>% summarise(Count=n()) %>% 
  spread(MAR_text,Count) 
tmp_mar <- apply(tmp_mar,2,function(x){as.numeric(x)*100/sum(as.numeric(x))})  
tmp_mar %>% as.data.frame() %>% gather("MARRIAGE","VALUE",-DEFAULT) %>% 
  mutate(DEFAULT=ifelse(DEFAULT==100,"YES","NO")) %>%
  mutate(per=comma(VALUE,digits=2)) %>%
  mutate(per=paste(per,"%")) %>%
  ggplot(aes(MARRIAGE,VALUE, fill=DEFAULT)) + 
  geom_bar(position = "stack", stat = "identity") +
  geom_text(aes(y=VALUE, label=per), color="white", size=3, vjust=1) +
  theme_minimal() +
  scale_fill_manual(values=c("blue", "red")) +
  ggtitle("PERCENTAGE(%) OF DEFAULT BY MARRIAGE") + ylab("PERCENTAGE(%)") +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "top")

#---------------------------- . ----------------------------#
#    AGE_FAC vs DEFAULT
#---------------------------- . ----------------------------#
dat_raw %>%  ggplot(aes(AGE_FAC,fill=DEF_text)) + geom_bar() +
  ggtitle("COUNT OF DEFAULT BY AGE(FACTORIZED)") + xlab("AGE(FACTORIZED)") + ylab("Count") +
  theme_minimal() + theme(plot.title = element_text(hjust = 0.5)) + 
  scale_fill_manual(name="DEFAULT", values=c("blue", "red"))

tmp_age <- dat_raw %>% group_by(AGE_FAC,DEFAULT) %>% summarise(Count=n()) %>% 
  spread(AGE_FAC,Count) 
tmp_age <- apply(tmp_age,2,function(x){as.numeric(x)*100/sum(as.numeric(x))})  
tmp_age %>% as.data.frame() %>% gather("AGE","VALUE",-DEFAULT) %>% 
  mutate(DEFAULT=ifelse(DEFAULT==100,"YES","NO")) %>%
  mutate(per=comma(VALUE,digits=2)) %>%
  mutate(per=paste(per,"%")) %>%
  ggplot(aes(AGE,VALUE, fill=DEFAULT)) + 
  geom_bar(position = "stack", stat = "identity") +
  geom_text(aes(y=VALUE, label=per), color="white", size=3, vjust=1) +
  theme_minimal() +
  scale_fill_manual(values=c("blue", "red")) +
  ggtitle("PERCENTAGE(%) OF DEFAULT BY AGE(FACTORIZED)") + ylab("PERCENTAGE(%)") +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "top")


#---------------------------- . ----------------------------#
#    PAY_1 vs DEFAULT
#      DEFAULT, does not happens if the client paid before due date
#---------------------------- . ----------------------------#

# What does default.payment.next.month means here, 
# and (in definition) relationship with PAY_0
# if PAY_0 = -1, means pay on time, so the default here is happens after 
#   the PAY_0-th months after Sep 2005.
# According to Wikipedia: 
#   To default is to fail to make a payment on a debt by the due date. 
#   If this happens with a credit card, creditors might raise interest rates to the default (or penalty rate) or decrease the line of credit. In case of serious delinquency, the card issuer can even take legal action to enforce payment or to garnish wages

# If client default on Sep, doesnot means will default Oct, vice-versa.
# so, data is correct when DEFAULT=1 but PAY_1=-1.
# The more interesting data is PAY_1 > 1, means payment for Sep is after Oct. 
dat_raw %>% select(DEF_text,PAY_1,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6) %>% gather(PAY,GAP,-DEF_text) %>%
  ggplot(aes(GAP)) + geom_bar(aes(label=GAP,fill=DEF_text)) + 
  theme_minimal() +
  scale_fill_manual(values=c("blue", "red"), name="DEFAULT") +
  ggtitle("DEFAULT BY PAYMENT PERIOD") + ylab("Count") + xlab("PAYMENT PERIOD") +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "top")

dat_raw %>% select(DEF_text,PAY_1,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6) %>% gather(PAY,GAP,-DEF_text) %>%
  ggplot(aes(GAP)) + geom_bar(aes(label=GAP,fill=DEF_text)) + 
  theme_minimal() +
  scale_fill_manual(values=c("blue", "red"), name="DEFAULT") +
  ggtitle("DEFAULT BY PAYMENT PERIOD") + ylab("Count") + xlab("PAYMENT PERIOD") +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "top") +
  facet_grid(~PAY)

# REMOVE those paying on time
dat_raw %>% select(DEF_text,PAY_1,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6) %>% gather(PAY,GAP,-DEF_text) %>%
  filter(GAP>0) %>%
  ggplot(aes(GAP)) + geom_bar(aes(label=GAP,fill=DEF_text)) + 
  theme_minimal() +
  scale_fill_manual(values=c("blue", "red"), name="DEFAULT") +
  ggtitle("DEFAULT BY PAYMENT PERIOD (REMOVED PAID ON TIME)") + ylab("Count") + xlab("PAYMENT PERIOD") +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "top") 

dat_raw %>% select(DEF_text,PAY_1,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6) %>% gather(PAY,GAP,-DEF_text) %>%
  filter(GAP>0) %>%
  ggplot(aes(GAP)) + geom_bar(aes(label=GAP,fill=DEF_text)) + 
  theme_minimal() +
  scale_fill_manual(values=c("blue", "red"), name="DEFAULT") +
  ggtitle("DEFAULT BY PAYMENT PERIOD (REMOVED PAID ON TIME)") + ylab("Count") + xlab("PAYMENT PERIOD") +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "top") +
  facet_grid(~PAY)

dat_raw %>% select(DEF_text,PAY_1,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6) %>% gather(PAY,GAP,-DEF_text) %>%
  filter(GAP>0) %>%
  ggplot(aes(PAY)) + geom_bar(aes(label=GAP,fill=DEF_text)) + 
  theme_minimal() +
  scale_fill_manual(values=c("blue", "red"), name="DEFAULT") +
  ggtitle("DEFAULT BY PAYMENT PERIOD (REMOVED PAID ON TIME)") + ylab("Count") + xlab("PAYMENT PERIOD") +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "top") +
  facet_grid(~GAP)


#---------------------------- . ----------------------------#
#    PAY_1, PAY_2, PAY_3, PAY_4, PAY_5 & PAY_6 
#    - Introduce PAYON_1 . PAYON_6, to indicate if payment is made before or after Oct.
#    - if PAY_1 > 1, means payment is after Oct, thus PAYON_1 = 1
#    - if PAY_2 > 2, means payment is after Oct, thus PAYON_2 = 1
#    - if each payment is made after Sep, then there is high probability of DEFAULT OCT.
#---------------------------- . ----------------------------#
dat_raw <- dat_raw %>% mutate(PAYON_1 = ifelse(PAY_1>1,1,0),
                              PAYON_2 = ifelse(PAY_2>2,1,0),
                              PAYON_3 = ifelse(PAY_3>3,1,0),
                              PAYON_4 = ifelse(PAY_4>4,1,0),
                              PAYON_5 = ifelse(PAY_5>5,1,0),
                              PAYON_6 = ifelse(PAY_6>6,1,0))

#tmp_categorical <- c('PAYON_1','PAYON_2','PAYON_3','PAYON_4','PAYON_5','PAYON_6')
#dat_raw[tmp_categorical] <- lapply(dat_raw[tmp_categorical], function(x) as.factor(x))

dat_payon <- dat_raw %>% select(DEFAULT,PAYON_1,PAYON_2,PAYON_3,PAYON_4,PAYON_5,PAYON_6) %>%
  gather("PAYON","value", -DEFAULT) %>% filter(value==1) %>%
  mutate(DEFAULT=ifelse(DEFAULT==1,"YES","NO")) 

dat_payon %>%
  ggplot(aes(PAYON,fill=DEFAULT), ) + geom_bar() +
  xlab("Payment Made After OCT") + ylab("Count") + ggtitle("DEFAULT DISTRIBUTION ON PAYMENT MADE AFTER OCT") +
  scale_fill_manual(values=c("blue", "red", "green", "yellow", "pink", "purple", "orange")) + 
  theme_minimal() + theme(legend.position = "top", plot.title = element_text(hjust = 0.5))

# BASE on chart, number of Default pay after OCT is more on Payon_1, and less for other months. 
# This is valid assumption, as PAYON_2 and others has more months to pay for BILLED.
# However, count of PAYON_1 is a 6.4803 times compare with PAYON_2
sum(dat_payon$PAYON=="PAYON_1") /sum(dat_payon$PAYON=="PAYON_2") 

# The probability of default vs Pay on time for Oct is higher, if a client default on a month:
# also, the default probability is higher, if client yet to pay for historical default.
dat_payon_sum <- dat_payon %>% 
  group_by(PAYON,DEFAULT) %>% summarise(count=n()) %>% spread (DEFAULT,count) %>%
  mutate(MTH_DEFAULT=YES+NO) %>% mutate(PROB_DEFAULT=YES/MTH_DEFAULT)

dat_payon_sum
dat_payon_sum %>% ggplot(aes(PAYON,PROB_DEFAULT)) + geom_point(col="red",size=3) +
  xlab("MONTH") + ylab("PERCENTAGE") + ggtitle("PROBABILITY OF DEFAULT ON PAYMENT MADE AFTER OCT") +
  theme_minimal() + theme(plot.title = element_text(hjust = 0.5))


#---------------------------- . ----------------------------#
#    INTRODUCE PAYDUE COLUMN
#    - PAYDUE column is derive from the probability of client not paying
#      OCT month due on time, based on PAYON.
#    - weightage from PAYON is use to value each PAYON contribution to overall
#   
#---------------------------- . ----------------------------#

tmp_PAYDUE <- dat_payon_sum$PROB_DEFAULT/(sum(dat_payon_sum$PROB_DEFAULT))
dat_raw <- dat_raw %>% mutate(PAYDUE=(as.numeric(PAYON_1)*tmp_PAYDUE[1])+
                                (PAYON_2*tmp_PAYDUE[2])+
                                (PAYON_3*tmp_PAYDUE[3])+
                                (PAYON_4*tmp_PAYDUE[4])+
                                (PAYON_5*tmp_PAYDUE[5])+
                                (PAYON_6*tmp_PAYDUE[6]))

dat_raw %>% arrange(desc(PAYDUE))

dat_raw %>% # filter(PAYDUE>0) %>%
  ggplot(aes(DEF_text,PAYDUE)) + geom_violin() +
  ylim(0,1) +
  ggtitle("PAYDUE DISTRIBUTION") + xlab("DEFAULT") + 
  theme_minimal() + theme(plot.title = element_text(hjust = 0.5)) 


dat_raw %>% # filter(PAYDUE>0) %>%
  ggplot(aes(DEF_text,PAYDUE)) + geom_violin() +
  ylim(0,1) +
  ggtitle("PAYDUE DISTRIBUTION") + xlab("DEFAULT") + 
  theme_minimal() + theme(plot.title = element_text(hjust = 0.5)) +
  facet_wrap(~EDU_text+SEX_text+MAR_text)


#---------------------------- . ----------------------------#
#    Correlation, Principal Components Analysis 
#    ON NUMERICAL FIELDS
#---------------------------- . ----------------------------#
dat_raw_sweep <- dat_raw %>% mutate(DEFAULT_N=as.numeric(DEFAULT)) %>%
  select("DEFAULT_N", "AGE_FAC", "PAYDUE", "LIMIT_BAL", 
         "BILLED_1", "BILLED_2", "BILLED_3",
         "BILLED_4", "BILLED_5", "BILLED_6",
         "PAYMENT_1", "PAYMENT_2", "PAYMENT_3",
         "PAYMENT_4", "PAYMENT_5", "PAYMENT_6") 

dat_raw_sweep <- sweep(dat_raw_sweep, 2, colMeans(dat_raw_sweep))

# Correlation Analysis
tmp_cor <- cor(dat_raw_sweep)
corrplot.mixed(
  tmp_cor,
  upper = "shade",
  lower = "number",
  tl.pos = "lt",
  tl.col = "black", 
  addCoef.col = "black",
  number.cex = .6,
  is.corr = TRUE
)

tmp_cor[1,]
# From Corr analysis, DEFAULT is associate with PAYDUE, and loosely with others (with reverse effect)


#------------- CLEANING ---------------
rm(tmp_age,tmp_edu,tmp_mar,tmp_PAYDUE,tmp_sex)
rm(dat_payon,dat_payon_sum)



############################# . #############################
#
#    PREPARE TRAINING & TEST DATA SET
#                 80% : 20%
############################# . #############################

set.seed(1, sample.kind = "Rounding")
tmp_idx_test <- createDataPartition(dat_raw$DEFAULT,
                                    times=1,p=0.2,list=FALSE)
dat_test <- dat_raw[tmp_idx_test,]
dat_train <- dat_raw[-tmp_idx_test,]

nrow(dat_train)
# 23999

nrow(dat_test)
# 6001

# Remove temporary variables
rm(tmp_idx_test)

# REMOVE VARIABLES TO CONSERVE MEMORY
rm(dat_raw,dat_raw_original,dat_raw_sweep,tmp_cor)


############################# . #############################
#
#    PERFORM MODELING
#                 
############################# . #############################

dat_train_simple <- dat_train %>% 
  select(DEFAULT, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE_FAC, PAYDUE, 
         PAYON_1, PAYON_2, PAYON_3, PAYON_4, PAYON_5, PAYON_6)

dat_test_simple <- dat_test %>% 
  select(DEFAULT, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE_FAC, PAYDUE, 
         PAYON_1, PAYON_2, PAYON_3, PAYON_4, PAYON_5, PAYON_6)

result <- tibble(Method = "", 
                 Accuracy = 0,
                 Balanced = 0,
                 Sensitivity = 0,
                 Specificity = 0,
                 F1 = 0)

f_printCM <- function(tmp_cm_table){
  rownames(tmp_cm_table) <- c("Predicted Positive(1)","Predicted Negative(0)")
  colnames(tmp_cm_table) <- c("Actual Positive(1)","Actual Negative(0)")
  tmp_cm_table  
}

#---------------------------- . ----------------------------#
#    MODELING USING Generalized Linear Model
#---------------------------- . ----------------------------#

glm_fit <- train(data=dat_train_simple, DEFAULT~.,method = "glm")
glm_predicted <- predict(glm_fit, newdata = dat_test)
glm_cm <- confusionMatrix(factor(glm_predicted),factor(dat_test$DEFAULT))

result <- rbind(result, with(glm_cm, 
                             c(Method = "GLM",
                               Accuracy = overall["Accuracy"],
                               Balanced = byClass["Balanced Accuracy"],
                               Sensitivity = byClass["Sensitivity"],
                               Specificity = byClass["Specificity"],
                               F1 = byClass["F1"])))

with(glm_cm, c(Method = "GLM",
               overall["Accuracy"],
               byClass["Balanced Accuracy"],
               byClass["Sensitivity"],
               byClass["Specificity"],
               byClass["F1"])) %>%
  knitr::kable(caption="PREDICTION RESULT USING GLM", digits=4)


f_printCM(glm_cm$table) 


#---------------------------- . ----------------------------#
#    MODELING USING  Linear Model
#---------------------------- . ----------------------------#

lm_fit <- lm(DEFAULT~.,
             data=dat_train_simple)
lm_predict <- predict(lm_fit, dat_test)
lm_predicted <- ifelse(lm_predict>1.5,1,0 )
lm_cm <- confusionMatrix(factor(lm_predicted),factor(dat_test$DEFAULT))
result <- rbind(result, with(lm_cm, 
                             c(Method = "LM",
                               Accuracy = overall["Accuracy"],
                               Balanced = byClass["Balanced Accuracy"],
                               Sensitivity = byClass["Sensitivity"],
                               Specificity = byClass["Specificity"],
                               F1 = byClass["F1"])))

with(lm_cm, c(Method = "LM",
              overall["Accuracy"],
              byClass["Balanced Accuracy"],
              byClass["Sensitivity"],
              byClass["Specificity"],
              byClass["F1"])) %>%
  knitr::kable(caption="PREDICTION RESULT USING LM", digits=4)

f_printCM(lm_cm$table) 

#---------------------------- . ----------------------------#
#    MODELING USING Generalized Additive Model using LOESS
#---------------------------- . ----------------------------#

loess_fit <- train(data=dat_train_simple, DEFAULT~., method = "gamLoess")
loess_predicted <- predict(loess_fit, dat_test)
loess_cm <- confusionMatrix(factor(loess_predicted),factor(dat_test$DEFAULT))

result <- rbind(result, with(loess_cm, 
                             c(Method = "LOESS",
                               Accuracy = overall["Accuracy"],
                               Balanced = byClass["Balanced Accuracy"],
                               Sensitivity = byClass["Sensitivity"],
                               Specificity = byClass["Specificity"],
                               F1 = byClass["F1"])))

with(loess_cm, c(Method = "LOESS",
                 overall["Accuracy"],
                 byClass["Balanced Accuracy"],
                 byClass["Sensitivity"],
                 byClass["Specificity"],
                 byClass["F1"])) %>%
  knitr::kable(caption="PREDICTION RESULT USING GAMLOESS", digits=4)

f_printCM(loess_cm$table)

#---------------------------- . ----------------------------#
#    MODELING USING K-nearest Neighbors
#---------------------------- . ----------------------------#

set.seed(1,sample.kind = "Rounding")
knn_control <- trainControl(method = "repeatedcv", number = 10, p = .9, repeats=10)
knn_fit <- train(data=dat_train_simple, DEFAULT~., method = "knn",
                 tuneGrid = data.frame(k = seq(3, 21, by=2)),
                 trControl=knn_control)
knn_predicted <- predict(knn_fit, dat_test)
knn_cm <- confusionMatrix(factor(knn_predicted),factor(dat_test$DEFAULT))

result <- rbind(result, with(knn_cm, 
                             c(Method = "K-NN",
                               Accuracy = overall["Accuracy"],
                               Balanced = byClass["Balanced Accuracy"],
                               Sensitivity = byClass["Sensitivity"],
                               Specificity = byClass["Specificity"],
                               F1 = byClass["F1"])))

with(knn_cm, c(Method = "K-NN",
               overall["Accuracy"],
               byClass["Balanced Accuracy"],
               byClass["Sensitivity"],
               byClass["Specificity"],
               byClass["F1"])) %>%
  knitr::kable(caption="PREDICTION RESULT USING K-NN", digits=4)

f_printCM(knn_cm$table)


#---------------------------- . ----------------------------#
#    MODELING USING K-Means Clustering
#---------------------------- . ----------------------------#

result <- rbind(result,  c(Method = "K-Means", Accuracy = 0.241959673387769, Balanced = 0.505182456769084,
                           Sensitivity = 0.0329552749839504, Specificity = 0.977409638554217, F1 = 0.0634136298126416))

set.seed(1, sample.kind = "Rounding")
f_predictkmeans <- function(x, k) {
  centers <- k$centers    # extract cluster centers
  # calculate distance to cluster centers
  distances <- sapply(1:nrow(x), function(i){
    apply(centers, 1, function(y) dist(rbind(x[i,], y)))
  })
  max.col(-t(distances))  # select cluster with min distance to center
}
k_means <- kmeans(dat_train_simple[,-1],centers=2)
k_predict <- f_predictkmeans(dat_test,k_means)
k_predicted <- ifelse(k_predict>1.5,0,1)
k_cm <- confusionMatrix(factor(k_predicted),factor(dat_test$DEFAULT))

result <- rbind(result, with(k_cm, 
                             c(Method = "K-Means",
                               Accuracy = overall["Accuracy"],
                               Balanced = byClass["Balanced Accuracy"],
                               Sensitivity = byClass["Sensitivity"],
                               Specificity = byClass["Specificity"],
                               F1 = byClass["F1"])))


with(k_cm, c(Method = "K-Means",
             overall["Accuracy"],
             byClass["Balanced Accuracy"],
             byClass["Sensitivity"],
             byClass["Specificity"],
             byClass["F1"])) %>%
  knitr::kable(caption="PREDICTION RESULT USING K-Means", digits=4)

f_printCM(k_cm$table)


#---------------------------- . ----------------------------#
#    MODELING USING Random Forest
#---------------------------- . ----------------------------#

#Search for the optimal value (with respect to Out-of-Bag error estimate) of mtry: 

set.seed(1,sample.kind = "Rounding")

rf_ntrees <- seq(200,by=200, len=10)
rf_tuning <- sapply(rf_ntrees, function(nt){
  t_mtry <- tuneRF(dat_train_simple[,-1],
                   dat_train_simple$DEFAULT,
                   stepFactor=1.5, improve=1e-5, ntree=nt,
                   plot=TRUE,trace=FALSE)
  t_mtry
})

rf_tuningT <- sapply(rf_tuning, function(rf){rf[1:3,]})[4:6,]
rf_tuningT <- cbind(c(2,3,4),rf_tuningT)
rownames(rf_tuningT) <- c(2,3,4)
colnames(rf_tuningT) <- c("mtry",rf_ntrees)

rf_mtree_optimal <- as.numeric(names(which.min(rf_tuningT[2,])))
rf_mtry <- c(2,3,4,5,8,10,20,50)

rf_tuningT %>% as.data.frame() %>% gather("ntree","value",-mtry) %>%
  mutate(ntree=str_sub(paste0("000",ntree),-6,-1)) %>%
  mutate(mtry=as.factor(mtry)) %>%
  ggplot(aes(mtry,value)) + geom_point() +
  ggtitle("RANDOM FOREST MTRY & NTREE PARAMETERS TUNNING") +
  ylab("OOB ERROR") +
  geom_hline(yintercept = 0.18125,col="blue", linetype="dashed", size=0.5) +
  geom_vline(xintercept = 2,col="blue", linetype="dashed", size=0.5) +
  theme_minimal() + theme(plot.title = element_text(hjust = 0.5)) +
  facet_wrap(~ntree, strip.position="bottom")

# Rerun the simulation with mtry value

rf_tuneGrid <- expand.grid(.mtry = rf_mtry)
rf_metric <- "Accuracy"
rf_control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")

rf_fit <- randomForest(data=dat_train_simple, DEFAULT~.,
                       ntree=rf_mtree_optimal,
                       metric=rf_metric, 
                       tuneGrid=rf_tuneGrid, 
                       trControl=rf_control)

rf_predicted <- predict(rf_fit, dat_test)
rf_cm <- confusionMatrix(factor(rf_predicted),factor(dat_test$DEFAULT))

rf_mtry_optimal <- rf_fit$mtry

result <- rbind(result, with(rf_cm, 
                             c(Method = paste0("RF(",rf_mtry_optimal,",",rf_mtree_optimal,")"),
                               Accuracy = overall["Accuracy"],
                               Balanced = byClass["Balanced Accuracy"],
                               Sensitivity = byClass["Sensitivity"],
                               Specificity = byClass["Specificity"],
                               F1 = byClass["F1"])))

with(rf_cm, c(Method = "RandomForest",
              overall["Accuracy"],
              byClass["Balanced Accuracy"],
              byClass["Sensitivity"],
              byClass["Specificity"],
              byClass["F1"])) %>%
  knitr::kable(caption="PREDICTION RESULT USING RANDOMFOREST", digits=4)


f_printCM(rf_cm$table)

importance(rf_fit)%>% as.data.frame() %>%
  arrange(-MeanDecreaseGini) %>% 
  knitr::kable(caption="MeanDecreaseGini")



#---------------------------- . ----------------------------#
#    MODELING USING Gradient Boosting
#---------------------------- . ----------------------------#

dat_train_simple_x <- dat_train_simple %>% select(-DEFAULT) %>% data.matrix()
dat_train_simple_y <- dat_train_simple %>% 
  mutate(DEFAULT=as.numeric(DEFAULT)-1) %>%
  select(DEFAULT) %>% data.matrix()


dat_test_simple_x <- dat_test_simple %>% select(-DEFAULT) %>% data.matrix()
dat_test_simple_y <- dat_test_simple %>% 
  mutate(DEFAULT=as.numeric(DEFAULT)-1) %>%
  select(DEFAULT) %>% data.matrix()

# PERFORM XGBOOST MODELING
dat_train_m <- xgb.DMatrix(data = dat_train_simple_x, label= dat_train_simple_y)
dat_test_m <- xgb.DMatrix(data = dat_test_simple_x, label= dat_test_simple_y)

xg_fit <- xgboost(data = dat_train_m, # the data   
                  max.depth = 5, # the maximum depth of each decision tree
                  nround = 1000, # number of boosting rounds
                  early_stopping_rounds = 10, # stop, if no improvement after rounds 
                  gamma = 1, # add a regularization term
                  objective = "binary:logistic",
                  verbose=0) 

xg_predicted <- predict(xg_fit, dat_test_m)
xg_cm <- confusionMatrix(factor(ifelse(xg_predicted>0.5,1,0)),factor(dat_test_simple_y))

result <- rbind(result, with(xg_cm, 
                             c(Method = "XGBOOST",
                               Accuracy = overall["Accuracy"],
                               Balanced = byClass["Balanced Accuracy"],
                               Sensitivity = byClass["Sensitivity"],
                               Specificity = byClass["Specificity"],
                               F1 = byClass["F1"])))

with(xg_cm, c(Method = "XGBOOST",
              overall["Accuracy"],
              byClass["Balanced Accuracy"],
              byClass["Sensitivity"],
              byClass["Specificity"],
              byClass["F1"])) %>%
  knitr::kable(caption="PREDICTION RESULT USING XGBOOST", digits=4)


xg_fit$evaluation_log %>% ggplot(aes(iter,train_error)) + 
  geom_line() + geom_point(size=2,col="red",shape=22) +
  theme_minimal() +
  ggtitle("XG Boost Train Error Calculation") + 
  ylab("Train Error") + xlab("Iteration") +
  theme(plot.title = element_text(hjust = 0.5))


f_printCM(xg_cm$table)


xgb.importance(model=xg_fit)%>% as.data.frame() %>% 
  knitr::kable(caption="XGBoost Variables Importancy", digits=4)


############################# . #############################
#
#    Intepreting Result
#    
############################# . #############################

result <- as.matrix(result)
result <- result[-1,]


result %>% as.data.frame() %>% 
  mutate(Accuracy=as.numeric(Accuracy)) %>%
  mutate(Balanced=as.numeric(Balanced)) %>%
  mutate(Sensitivity=as.numeric(Sensitivity)) %>%
  mutate(Specificity=as.numeric(Specificity)) %>%
  mutate(F1=as.numeric(F1)) %>%
  knitr::kable(caption="PREDICTION RESULT", digits=4, format.args=list(big.mark=","))

#---------------------------- . ----------------------------#
#    MODELING USING COMPREHENSIVE DATASET
#---------------------------- . ----------------------------#

nzv <- nearZeroVar(dat_train)
dat_train_more <- dat_train %>% 
  select(-all_of(nzv)) %>%
  select(-c(EDU_text, SEX_text, MAR_text, DEF_text)) %>%
  select(-AGE)

names(dat_train_more)

#---------------------------- . ----------------------------#
#    COMPREHENSIVE DATASET MODELING RESULT
#---------------------------- . ----------------------------#

# Following results are executed separately and NOT re-execute here. 
# The result can be obtained by executing the script as in Appendix A
f_result <- function(text){
  tmp <- sapply(strsplit(text,'|',fixed=TRUE) [[1]],function(x){str_squish(str_trim(x))})[-1]
  tmp <- t(matrix(tmp,7,ceiling(length(tmp)/7)) [,c(-2,-3)])[,-7]
  colnames(tmp)<- tmp[1,]
  tmp <- tmp[-1,]
  tmp[,2:6] <- as.numeric(tmp[,2:6])
  tmp
}

result_more <- f_result("|Method    |Accuracy          |Balanced          |Sensitivity        |Specificity        |F1                 |
|:---------|:-----------------|:-----------------|:------------------|:------------------|:------------------|
|          |0                 |0                 |0                  |0                  |0                  |
|GLM       |0.823029495084153 |0.650548669103979 |0.959982880376632  |0.341114457831325  |0.894159856487941  |
|LM        |0.824029328445259 |0.654963772272914 |0.958270918039803  |0.351656626506024  |0.894526568118258  |
|LOESS     |0.822529578403599 |0.648071609141466 |0.96105285683715   |0.335090361445783  |0.893998208420424  |
|K-NN      |0.775704049325112 |0.532571114760776 |0.968756687352878  |0.0963855421686747 |0.870576923076923  |
|K-Means   |0.242959506748875 |0.500973291840592 |0.0380911619944361 |0.963855421686747  |0.0726678914064095 |
|RF(3,800) |0.82102982836194  |0.656810931936606 |0.951423068692489  |0.362198795180723  |0.892233594220349  |
|XGBOOST   |0.82102982836194  |0.660853557607275 |0.948213139310935  |0.373493975903614  |0.891908212560387  |") 
result_more %>% as.data.frame() %>% 
  mutate(Accuracy=as.numeric(Accuracy)) %>%
  mutate(Balanced=as.numeric(Balanced)) %>%
  mutate(Sensitivity=as.numeric(Sensitivity)) %>%
  mutate(Specificity=as.numeric(Specificity)) %>%
  mutate(F1=as.numeric(F1)) %>%
  knitr::kable(caption="PREDICTION RESULT (COMPREHENSIVE DATASET)", digits=4, format.args=list(big.mark=","))

#---------------------------- . ----------------------------#
#    COMPARE RESULT BETWEEN SIMPLIFIED & COMPREHENSIVE DATASET
#---------------------------- . ----------------------------#

tmp_result_compare <- cbind(result[,1],matrix(as.numeric(result_more[,2:6])-as.numeric(result[,2:6]),7,5))
colnames(tmp_result_compare) <- colnames(result)
tmp_result_compare %>% as.data.frame() %>% 
  mutate(DiffAccuracy       =round(as.numeric(Accuracy),4)) %>%
  mutate(DiffBalanced       =round(as.numeric(Balanced),4)) %>%
  mutate(DiffSensitivity    =round(as.numeric(Sensitivity),4)) %>%
  mutate(DiffSpecificity    =round(as.numeric(Specificity),4)) %>%
  mutate(DiffF1             =round(as.numeric(F1),4)) %>%
  mutate(Accuracy           =round(as.numeric(result_more[,2]),4)) %>%
  mutate(Balanced           =round(as.numeric(result_more[,3]),4)) %>%
  mutate(Sensitivity        =round(as.numeric(result_more[,4]),4)) %>%
  mutate(Specificity        =round(as.numeric(result_more[,5]),4)) %>%
  mutate(F1                 =round(as.numeric(result_more[,6]),4)) %>% t() %>%
  knitr::kable(caption="COMPARE DATASET PREDICTION RESULT", digits=4, format.args=list(big.mark=","))


#---------------------------- . ----------------------------#
#    WHICH IS THE BEST F1?
#---------------------------- . ----------------------------#

result %>% as.data.frame() %>% 
  mutate(Accuracy=as.numeric(Accuracy)) %>%
  mutate(Balanced=as.numeric(Balanced)) %>%
  mutate(Sensitivity=as.numeric(Sensitivity)) %>%
  mutate(Specificity=as.numeric(Specificity)) %>%
  mutate(F1=as.numeric(F1)) %>%
  knitr::kable(caption="PREDICTION RESULT", digits=4, format.args=list(big.mark=","))

result %>% as.data.frame() %>% gather("Metrix","Value",-Method) %>%
  mutate(Value=as.numeric(Value)) %>%
  ggplot(aes(Metrix, Value,col=Method)) + geom_point(aes(shape=Method), size=3) +
  scale_shape_manual(values = c(15:25)) +
  theme_minimal() +
  ggtitle("Method Evaluation") + 
  theme(plot.title = element_text(hjust = 0.5))

# conclude that GLM method  is the best method

#---------------------------- . ----------------------------#
#    MODELING USING OTHER METHOD
#---------------------------- . ----------------------------#
# + Naive_bayes
# + SVMLinear
# + GAMBoost
# + kknn
# + GAM
# + Ranger
# + WSRF
# + Rborist
# + avNNet
# + MLP
# + monMLP
# + GBM
# + ADABoost
# + SVMRadial
# + SVMRadialCost
# + SVMRadialSigma

# result_other is performed outside of RMARKDOWN as the the modeling processes take more than 4 hours.
# refer to APPENDIX B, and execute if necessary.

result_other <- f_result(" |Method      |Accuracy          |Balanced          |Sensitivity       |Specificity       |F1                |
|:-----------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
|            |0                 |0                 |0                 |0                 |0                 |
|naive_bayes |0.823529411764706 |0.653025729066491 |0.958912903916114 |0.347138554216867 |0.894321923959685 |
|svmLinear   |0.623062822862856 |0.58872006321885  |0.650331692702761 |0.52710843373494  |0.728776978417266 |
|kknn        |0.78770204965839  |0.619240900043573 |0.921463727797988 |0.317018072289157 |0.871130892170747 |
|gam         |0.823529411764706 |0.650061136908    |0.961266852129253 |0.338855421686747 |0.894553420292741 |
|ranger      |0.823362772871188 |0.650493156018037 |0.960624866252942 |0.340361445783133 |0.894401275154413 |
|wsrf        |0.818863522746209 |0.632781258782186 |0.966616734431842 |0.29894578313253  |0.892599545499457 |")

rbind(result[which.max(result[,6]),], result_other) %>% 
  as.data.frame() %>% 
  mutate(Accuracy=as.numeric(Accuracy)) %>%
  mutate(Balanced=as.numeric(Balanced)) %>%
  mutate(Sensitivity=as.numeric(Sensitivity)) %>%
  mutate(Specificity=as.numeric(Specificity)) %>%
  mutate(F1=as.numeric(F1)) %>%
  knitr::kable(caption="PREDICTION RESULT (OTHER ALGORITHMS)", digits=4, format.args=list(big.mark=","))

rbind(result[which.max(result[,6]),], result_other)[,c(1,6)] %>% 
  as.data.frame() %>% mutate(F1=as.numeric(F1)) %>%
  ggplot(aes(Method, F1,col=Method)) + geom_point(aes(shape=Method), size=3) +
  geom_text(aes(label=comma(F1,digits=4)),check_overlap = TRUE, vjust=2) +
  scale_shape_manual(values = c(15:25)) +
  theme_minimal() +
  ggtitle("OTHER ALGORITHM EVALUATION") + 
  theme(plot.title = element_text(hjust = 0.5),legend.position = "na")


############################# . #############################
#
#    Appendix A: Modeling With "Comprehensive Dataset
#    
############################# . #############################
#............................ . ............................#
#    COMPREHENSIVE DATASET
#    - Preparing Script
#............................ . ............................#

sink(paste0("modeling-dat_train_more-",format(Sys.Date(),"%Y%m%d"),".log"), 
     append=FALSE, split=TRUE)

nzv <- nearZeroVar(dat_train)
dat_train_more <- dat_train %>% 
  select(-all_of(nzv)) %>%
  select(-c(EDU_text, SEX_text, MAR_text, DEF_text)) %>%
  select(-AGE)

dat_test_more <- dat_test %>% 
  select(-all_of(nzv)) %>%
  select(-c(EDU_text, SEX_text, MAR_text, DEF_text)) %>%
  select(-AGE)

#---------------------------- . ----------------------------#

result_more <- tibble(Method = "", 
                      Accuracy = 0,
                      Balanced = 0,
                      Sensitivity = 0,
                      Specificity = 0,
                      F1 = 0)

#---------------------------- . ----------------------------#
#    PREDICT USING LOGISTIC REGRESSION 
#---------------------------- . ----------------------------#

glm_fit <- train(data=dat_train_more, DEFAULT~.,method = "glm")
glm_predicted <- predict(glm_fit, newdata = dat_test)
glm_cm <- confusionMatrix(factor(glm_predicted),factor(dat_test$DEFAULT))

result_more <- rbind(result_more, with(glm_cm, 
                                       c(Method = "GLM",
                                         Accuracy = overall["Accuracy"],
                                         Balanced = byClass["Balanced Accuracy"],
                                         Sensitivity = byClass["Sensitivity"],
                                         Specificity = byClass["Specificity"],
                                         F1 = byClass["F1"])))
result_more %>% knitr::kable(caption="PREDICTION ACCURACY", digits=4)

#---------------------------- . ----------------------------#
#    PREDICT USING LM 
#---------------------------- . ----------------------------#
lm_fit <- lm(DEFAULT~.,
             data=dat_train_more)
lm_predict <- predict(lm_fit, dat_test)
lm_predicted <- ifelse(lm_predict>1.5,1,0 )
lm_cm <- confusionMatrix(factor(lm_predicted),factor(dat_test$DEFAULT))

result_more <- rbind(result_more, with(lm_cm, 
                                       c(Method = "LM",
                                         Accuracy = overall["Accuracy"],
                                         Balanced = byClass["Balanced Accuracy"],
                                         Sensitivity = byClass["Sensitivity"],
                                         Specificity = byClass["Specificity"],
                                         F1 = byClass["F1"])))
result_more %>% knitr::kable(caption="PREDICTION ACCURACY", digits=4)

#---------------------------- . ----------------------------#
#    PREDICT LOESS 
#---------------------------- . ----------------------------#
loess_fit <- train(data=dat_train_more, DEFAULT~., method = "gamLoess")
loess_predicted <- predict(loess_fit, dat_test)
loess_cm <- confusionMatrix(factor(loess_predicted),factor(dat_test$DEFAULT))

result_more <- rbind(result_more, with(loess_cm, 
                                       c(Method = "LOESS",
                                         Accuracy = overall["Accuracy"],
                                         Balanced = byClass["Balanced Accuracy"],
                                         Sensitivity = byClass["Sensitivity"],
                                         Specificity = byClass["Specificity"],
                                         F1 = byClass["F1"])))
result_more %>% knitr::kable(caption="PREDICTION ACCURACY", digits=4)



#---------------------------- . ----------------------------#
#    PREDICT KNN 
#---------------------------- . ----------------------------#
set.seed(1,sample.kind = "Rounding")
knn_control <- trainControl(method = "repeatedcv", number = 10, p = .9, repeats=10)
knn_fit <- train(data=dat_train_more, DEFAULT~., method = "knn",
                 tuneGrid = data.frame(k = seq(3, 21, by=2)),
                 trControl=knn_control)
knn_predicted <- predict(knn_fit, dat_test)
knn_cm <- confusionMatrix(factor(knn_predicted),factor(dat_test$DEFAULT))

result_more <- rbind(result_more, with(knn_cm, 
                                       c(Method = "K-NN",
                                         Accuracy = overall["Accuracy"],
                                         Balanced = byClass["Balanced Accuracy"],
                                         Sensitivity = byClass["Sensitivity"],
                                         Specificity = byClass["Specificity"],
                                         F1 = byClass["F1"])))
result_more %>% knitr::kable(caption="PREDICTION ACCURACY", digits=4)


#---------------------------- . ----------------------------#
#    PREDICT K-MEANS 
#---------------------------- . ----------------------------#
set.seed(1, sample.kind = "Rounding")
f_predictkmeans <- function(x, k) {
  centers <- k$centers    # extract cluster centers
  # calculate distance to cluster centers
  distances <- sapply(1:nrow(x), function(i){
    apply(centers, 1, function(y) dist(rbind(x[i,], y)))
  })
  max.col(-t(distances))  # select cluster with min distance to center
}
k_means <- kmeans(dat_train_more[,-1],centers=2)
k_predict <- f_predictkmeans(dat_test,k_means)
k_predicted <- ifelse(k_predict>1.5,0,1)
k_cm <- confusionMatrix(factor(k_predicted),factor(dat_test$DEFAULT))

result_more <- rbind(result_more, with(k_cm, 
                                       c(Method = "K-Means",
                                         Accuracy = overall["Accuracy"],
                                         Balanced = byClass["Balanced Accuracy"],
                                         Sensitivity = byClass["Sensitivity"],
                                         Specificity = byClass["Specificity"],
                                         F1 = byClass["F1"])))
result_more %>% knitr::kable(caption="PREDICTION ACCURACY", digits=4)


#---------------------------- . ----------------------------#
#    PREDICT USING RANDOM FOREST 
#    - mtry: Number of variables randomly sampled as candidates at each split.
#    - ntree: Number of trees to grow.
#    - various mtry & ntree values combination are explored to reach the range as below 
#---------------------------- . ----------------------------#

# First, find the best parameter for RF
# 
# set.seed(1,sample.kind = "Rounding")
# 
# rf_ntrees <- seq(200,by=200, len=10)
# rf_tuning <- sapply(rf_ntrees, function(nt){
#   t_mtry <- tuneRF(dat_train_more[,-1], 
#                    dat_train_more$DEFAULT,
#                    stepFactor=1.5, improve=1e-5, ntree=nt, 
#                    plot=TRUE,trace=FALSE)
#   t_mtry
# })
# 
# 
# rf_tuningT <- cbind(rf_tuning[[x1]][1:3,],ntree=rep(3,rf_ntrees[x]))
# rep(2:10,function(x){
#     rf_tuningT <- rbind(rf_tuningT, cbind(rf_tuning[[x]][1:3,],ntree=rep(3,rf_ntrees[x])))
# })
# 
# x=5
# rf_tuningT <- rbind(rf_tuningT, cbind(rf_tuning[[x]][1:3,],ntree=rep(3,rf_ntrees[x])))
# rf_ntrees[4]
# rf_tuning
# rownames(rf_tuning) <- c(2,3,4)
# colnames(rf_tuning) <- rf_ntrees
# rf_tuning <- cbind(rf_tuning,mtry=c(2,3,4))
# 
# # t_mtry <- tuneRF(dat_train_more[,-1], 
# #                  dat_train_more$DEFAULT,
# #                  stepFactor=1.5, improve=1e-5, ntreeTry=1200, 
# #                  plot=TRUE,trace=TRUE,doBest=TRUE)
# 
# rf_tuning %>% as.data.frame() %>% gather("ntree","value",-mtry) %>%
#   mutate(ntree=str_sub(paste0("000",ntree),-6,-1)) %>%
#   mutate(mtry=as.factor(mtry)) %>%
#   ggplot(aes(mtry,value)) + geom_point() + 
#   ggtitle("RANDOM FOREST MTRY & NTREE PARAMETERS TUNNING") +
#   ylab("OOB ERROR") + 
#   geom_hline(yintercept = 0.18125,col="blue", linetype="dashed", size=0.5) +
#   geom_vline(xintercept = 2,col="blue", linetype="dashed", size=0.5) +
#   theme_minimal() + theme(plot.title = element_text(hjust = 0.5)) +
#   facet_wrap(~ntree, strip.position="bottom")

#----------------
# Based on above chart, as mtry=3 is more consistent as recommended, 
# I selected mtry=3, ntree=800, mtry=3 is further confirm with following 

rf_tuneGrid <- expand.grid(.mtry = c(2,3,4,5,8,10,20,50))
rf_metric <- "Accuracy"
rf_control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")

rf_fit <- randomForest(data=dat_train_more, DEFAULT~.,
                       ntree=800,
                       metric=rf_metric, 
                       tuneGrid=rf_tuneGrid, 
                       trControl=rf_control)

rf_predicted <- predict(rf_fit, dat_test)
rf_cm <- confusionMatrix(factor(rf_predicted),factor(dat_test$DEFAULT))

result_more <- rbind(result_more, with(rf_cm, 
                                       c(Method = "RF(3,800)",
                                         Accuracy = overall["Accuracy"],
                                         Balanced = byClass["Balanced Accuracy"],
                                         Sensitivity = byClass["Sensitivity"],
                                         Specificity = byClass["Specificity"],
                                         F1 = byClass["F1"])))
result_more %>% knitr::kable(caption="PREDICTION ACCURACY", digits=4)


# -----------------------------------------------------
# STUDY IMPORTANT of VARIABLE
importance(rf_fit)%>% as.data.frame() %>%
  arrange(-MeanDecreaseGini) %>% 
  knitr::kable(caption="MeanDecreaseGini")





#---------------------------- . ----------------------------#
#    PREDICT USING XGBOOST 
#---------------------------- . ----------------------------#

dat_train_more_x <- dat_train_more %>% select(-DEFAULT) %>% data.matrix()
dat_train_more_y <- dat_train_more %>% 
  mutate(DEFAULT=as.numeric(DEFAULT)-1) %>%
  select(DEFAULT) %>% data.matrix()


dat_test_more_x <- dat_test_more %>% select(-DEFAULT) %>% data.matrix()
dat_test_more_y <- dat_test_more %>% 
  mutate(DEFAULT=as.numeric(DEFAULT)-1) %>%
  select(DEFAULT) %>% data.matrix()

# PERFORM XGBOOST MODELING
dat_train_m <- xgb.DMatrix(data = dat_train_more_x, label= dat_train_more_y)
dat_test_m <- xgb.DMatrix(data = dat_test_more_x, label= dat_test_more_y)

xg_fit <- xgboost(data = dat_train_m, # the data   
                  max.depth = 5, # the maximum depth of each decision tree
                  nround = 1000, # number of boosting rounds
                  early_stopping_rounds = 10, # if we dont see an improvement in this many rounds, stop
                  #scale_pos_weight = sum(dat_train_simple_y==1)/sum(dat_train_simple_y==0), # control for imbalanced classes
                  gamma = 1, # add a regularization term
                  objective = "binary:logistic") 

xg_predicted <- predict(xg_fit, dat_test_m)
xg_cm <- confusionMatrix(factor(ifelse(xg_predicted>0.5,1,0)),factor(dat_test_more_y))

xgb.importance(model=xg_fit)


result_more <- rbind(result_more, with(xg_cm, 
                                       c(Method = "XGBOOST",
                                         Accuracy = overall["Accuracy"],
                                         Balanced = byClass["Balanced Accuracy"],
                                         Sensitivity = byClass["Sensitivity"],
                                         Specificity = byClass["Specificity"],
                                         F1 = byClass["F1"])))
result_more %>% knitr::kable(caption="PREDICTION RESULT", digits=4)

#---------------------------- . ----------------------------#
#    CONSOLIDATE PREDICTED RESULT 
#---------------------------- . ----------------------------#
result_more %>% knitr::kable(caption="PREDICTION RESULT", digits=4, format.args=list(big.mark=","))
sink()



############################# . #############################
#
#    Appendix B: "Modeling Using Other Methods"
#    
############################# . #############################

#............................ . ............................#
#    SIMPLIFIED DATASET
#............................ . ............................#

sink(paste0("modeling-dat_train-other-",format(Sys.Date(),"%Y%m%d"),".log"), 
     append=FALSE, split=TRUE)

# [1] "DEFAULT"   "LIMIT_BAL" "SEX"       "EDUCATION" "MARRIAGE"  "AGE_FAC"   "PAYDUE"    "PAYON_1"  
# [9] "PAYON_2"   "PAYON_3"   "PAYON_4"   "PAYON_5"   "PAYON_6" 

dat_train_simple <- dat_train %>% 
  select(DEFAULT, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE_FAC, PAYDUE, 
         PAYON_1, PAYON_2, PAYON_3, PAYON_4, PAYON_5, PAYON_6)

dat_test_simple <- dat_test %>% 
  select(DEFAULT, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE_FAC, PAYDUE, 
         PAYON_1, PAYON_2, PAYON_3, PAYON_4, PAYON_5, PAYON_6)

names(dat_train_simple)    #13 variables

#---------------------------- . ----------------------------#

result_other <- tibble(Method = "", 
                       Accuracy = 0,
                       Balanced = 0,
                       Sensitivity = 0,
                       Specificity = 0,
                       F1 = 0)


names(dat_train_simple)

############################# . #############################
#    MODELING USING OTHER APPROACH
#    - USE DEFAULT PARAMETER
#    - "naive_bayes", "svmLinear", "gamboost", "kknn", "loclda", "gam", 
#      "ranger","wsrf", "Rborist", "avNNet", "mlp", "monmlp", "gbm",
#      "adaboost", "svmRadial", "svmRadialCost", "svmRadialSigma",
#      "repeatedcv"
############################# . #############################


#---------------------------- . ----------------------------#
#    PREDICT USING naive_bayes
#---------------------------- . ----------------------------#
nb_fit <- train(data=dat_train_simple, DEFAULT~.,
                method = "naive_bayes")
nb_predicted <- predict(nb_fit, dat_test)
nb_cm <- confusionMatrix(factor(nb_predicted),factor(dat_test$DEFAULT))

result_other <- rbind(result_other, with(nb_cm, 
                                         c(Method = "naive_bayes",
                                           Accuracy = overall["Accuracy"],
                                           Balanced = byClass["Balanced Accuracy"],
                                           Sensitivity = byClass["Sensitivity"],
                                           Specificity = byClass["Specificity"],
                                           F1 = byClass["F1"])))
result_other %>% knitr::kable(caption="PREDICTION RESULT", digits=4)


#---------------------------- . ----------------------------#
#    PREDICT USING svmLinear
#---------------------------- . ----------------------------#
svml_fit <- train(data=dat_train_simple, DEFAULT~.,
                  method = "svmLinear")
svml_predicted <- predict(svml_fit, dat_test)
svml_cm <- confusionMatrix(factor(svml_predicted),factor(dat_test$DEFAULT))

result_other <- rbind(result_other, with(svml_cm, 
                                         c(Method = "svmLinear",
                                           Accuracy = overall["Accuracy"],
                                           Balanced = byClass["Balanced Accuracy"],
                                           Sensitivity = byClass["Sensitivity"],
                                           Specificity = byClass["Specificity"],
                                           F1 = byClass["F1"])))
result_other %>% knitr::kable(caption="PREDICTION RESULT", digits=4)


#---------------------------- . ----------------------------#
#    PREDICT USING gamboost
#---------------------------- . ----------------------------#
gb_fit <- train(data=dat_train_simple, DEFAULT~.,
                method = "gamboost")
gb_predicted <- predict(gb_fit, dat_test)
gb_cm <- confusionMatrix(factor(gb_predicted),factor(dat_test$DEFAULT))

result_other <- rbind(result_other, with(gb_cm, 
                                         c(Method = "gamboost",
                                           Accuracy = overall["Accuracy"],
                                           Balanced = byClass["Balanced Accuracy"],
                                           Sensitivity = byClass["Sensitivity"],
                                           Specificity = byClass["Specificity"],
                                           F1 = byClass["F1"])))
result_other %>% knitr::kable(caption="PREDICTION RESULT", digits=4)


#---------------------------- . ----------------------------#
#    PREDICT USING kknn
#---------------------------- . ----------------------------#
kknn_fit <- train(data=dat_train_simple, DEFAULT~.,
                  method = "kknn")
kknn_predicted <- predict(kknn_fit, dat_test)
kknn_cm <- confusionMatrix(factor(kknn_predicted),factor(dat_test$DEFAULT))

result_other <- rbind(result_other, with(kknn_cm, 
                                         c(Method = "kknn",
                                           Accuracy = overall["Accuracy"],
                                           Balanced = byClass["Balanced Accuracy"],
                                           Sensitivity = byClass["Sensitivity"],
                                           Specificity = byClass["Specificity"],
                                           F1 = byClass["F1"])))
result_other %>% knitr::kable(caption="PREDICTION RESULT", digits=4)


#---------------------------- . ----------------------------#
#    PREDICT USING gam
#---------------------------- . ----------------------------#
gam_fit <- train(data=dat_train_simple, DEFAULT~.,
                 method = "gam")
gam_predicted <- predict(gam_fit, dat_test)
gam_cm <- confusionMatrix(factor(gam_predicted),factor(dat_test$DEFAULT))

result_other <- rbind(result_other, with(gam_cm, 
                                         c(Method = "gam",
                                           Accuracy = overall["Accuracy"],
                                           Balanced = byClass["Balanced Accuracy"],
                                           Sensitivity = byClass["Sensitivity"],
                                           Specificity = byClass["Specificity"],
                                           F1 = byClass["F1"])))
result_other %>% knitr::kable(caption="PREDICTION RESULT", digits=4)


#---------------------------- . ----------------------------#
#    PREDICT USING ranger
#---------------------------- . ----------------------------#
ranger_fit <- train(data=dat_train_simple, DEFAULT~.,
                    method = "ranger")
ranger_predicted <- predict(ranger_fit, dat_test)
ranger_cm <- confusionMatrix(factor(ranger_predicted),factor(dat_test$DEFAULT))

result_other <- rbind(result_other, with(ranger_cm, 
                                         c(Method = "ranger",
                                           Accuracy = overall["Accuracy"],
                                           Balanced = byClass["Balanced Accuracy"],
                                           Sensitivity = byClass["Sensitivity"],
                                           Specificity = byClass["Specificity"],
                                           F1 = byClass["F1"])))
result_other %>% knitr::kable(caption="PREDICTION RESULT", digits=4)


#---------------------------- . ----------------------------#
#    PREDICT USING wsrf
#---------------------------- . ----------------------------#
wsrf_fit <- train(data=dat_train_simple, DEFAULT~.,
                  method = "wsrf")
wsrf_predicted <- predict(wsrf_fit, dat_test)
wsrf_cm <- confusionMatrix(factor(wsrf_predicted),factor(dat_test$DEFAULT))

result_other <- rbind(result_other, with(wsrf_cm, 
                                         c(Method = "wsrf",
                                           Accuracy = overall["Accuracy"],
                                           Balanced = byClass["Balanced Accuracy"],
                                           Sensitivity = byClass["Sensitivity"],
                                           Specificity = byClass["Specificity"],
                                           F1 = byClass["F1"])))
result_other %>% knitr::kable(caption="PREDICTION RESULT", digits=4)


#---------------------------- . ----------------------------#
#    PREDICT USING Rborist
#---------------------------- . ----------------------------#
Rbor_fit <- train(data=dat_train_simple, DEFAULT~.,
                  method = "Rborist")
Rbor_predicted <- predict(Rbor_fit, dat_test)
Rbor_cm <- confusionMatrix(factor(Rbor_predicted),factor(dat_test$DEFAULT))

result_other <- rbind(result_other, with(Rbor_cm, 
                                         c(Method = "Rborist",
                                           Accuracy = overall["Accuracy"],
                                           Balanced = byClass["Balanced Accuracy"],
                                           Sensitivity = byClass["Sensitivity"],
                                           Specificity = byClass["Specificity"],
                                           F1 = byClass["F1"])))
result_other %>% knitr::kable(caption="PREDICTION RESULT", digits=4)



#---------------------------- . ----------------------------#
#    PREDICT USING avNNet
#---------------------------- . ----------------------------#
avNNet_fit <- train(data=dat_train_simple, DEFAULT~.,
                    method = "avNNet")
avNNet_predicted <- predict(avNNet_fit, dat_test)
avNNet_cm <- confusionMatrix(factor(avNNet_predicted),factor(dat_test$DEFAULT))

result_other <- rbind(result_other, with(avNNet_cm, 
                                         c(Method = "avNNet",
                                           Accuracy = overall["Accuracy"],
                                           Balanced = byClass["Balanced Accuracy"],
                                           Sensitivity = byClass["Sensitivity"],
                                           Specificity = byClass["Specificity"],
                                           F1 = byClass["F1"])))
result_other %>% knitr::kable(caption="PREDICTION RESULT", digits=4)


#---------------------------- . ----------------------------#
#    PREDICT USING mlp
#---------------------------- . ----------------------------#
mlp_fit <- train(data=dat_train_simple, DEFAULT~.,
                 method = "mlp")
mlp_predicted <- predict(mlp_fit, dat_test)
mlp_cm <- confusionMatrix(factor(mlp_predicted),factor(dat_test$DEFAULT))

result_other <- rbind(result_other, with(mlp_cm, 
                                         c(Method = "mlp",
                                           Accuracy = overall["Accuracy"],
                                           Balanced = byClass["Balanced Accuracy"],
                                           Sensitivity = byClass["Sensitivity"],
                                           Specificity = byClass["Specificity"],
                                           F1 = byClass["F1"])))
result_other %>% knitr::kable(caption="PREDICTION RESULT", digits=4)


#---------------------------- . ----------------------------#
#    PREDICT USING monmlp
#---------------------------- . ----------------------------#
monmlp_fit <- train(data=dat_train_simple, DEFAULT~.,
                    method = "monmlp")
monmlp_predicted <- predict(monmlp_fit, dat_test)
monmlp_cm <- confusionMatrix(factor(monmlp_predicted),factor(dat_test$DEFAULT))

result_other <- rbind(result_other, with(monmlp_cm, 
                                         c(Method = "monmlp",
                                           Accuracy = overall["Accuracy"],
                                           Balanced = byClass["Balanced Accuracy"],
                                           Sensitivity = byClass["Sensitivity"],
                                           Specificity = byClass["Specificity"],
                                           F1 = byClass["F1"])))
result_other %>% knitr::kable(caption="PREDICTION RESULT", digits=4)


#---------------------------- . ----------------------------#
#    PREDICT USING gbm
#---------------------------- . ----------------------------#
gbm_fit <- train(data=dat_train_simple, DEFAULT~.,
                 method = "gbm")
gbm_predicted <- predict(gbm_fit, dat_test)
gbm_cm <- confusionMatrix(factor(gbm_predicted),factor(dat_test$DEFAULT))

result_other <- rbind(result_other, with(gbm_cm, 
                                         c(Method = "gbm",
                                           Accuracy = overall["Accuracy"],
                                           Balanced = byClass["Balanced Accuracy"],
                                           Sensitivity = byClass["Sensitivity"],
                                           Specificity = byClass["Specificity"],
                                           F1 = byClass["F1"])))
result_other %>% knitr::kable(caption="PREDICTION RESULT", digits=4)


#---------------------------- . ----------------------------#
#    PREDICT USING adaboost
#---------------------------- . ----------------------------#
adab_fit <- train(data=dat_train_simple, DEFAULT~.,
                  method = "adaboost")
adab_predicted <- predict(adab_fit, dat_test)
adab_cm <- confusionMatrix(factor(adab_predicted),factor(dat_test$DEFAULT))

result_other <- rbind(result_other, with(adab_cm, 
                                         c(Method = "adaboost",
                                           Accuracy = overall["Accuracy"],
                                           Balanced = byClass["Balanced Accuracy"],
                                           Sensitivity = byClass["Sensitivity"],
                                           Specificity = byClass["Specificity"],
                                           F1 = byClass["F1"])))
result_other %>% knitr::kable(caption="PREDICTION RESULT", digits=4)


#---------------------------- . ----------------------------#
#    PREDICT USING svmRadial
#---------------------------- . ----------------------------#
svmr_fit <- train(data=dat_train_simple, DEFAULT~.,
                  method = "svmRadial")
svmr_predicted <- predict(svmr_fit, dat_test)
svmr_cm <- confusionMatrix(factor(svmr_predicted),factor(dat_test$DEFAULT))

result_other <- rbind(result_other, with(svmr_cm, 
                                         c(Method = "svmRadial",
                                           Accuracy = overall["Accuracy"],
                                           Balanced = byClass["Balanced Accuracy"],
                                           Sensitivity = byClass["Sensitivity"],
                                           Specificity = byClass["Specificity"],
                                           F1 = byClass["F1"])))
result_other %>% knitr::kable(caption="PREDICTION RESULT", digits=4)


#---------------------------- . ----------------------------#
#    PREDICT USING svmRadialCost
#---------------------------- . ----------------------------#
svmrc_fit <- train(data=dat_train_simple, DEFAULT~.,
                   method = "svmRadialCost")
svmrc_predicted <- predict(svmrc_fit, dat_test)
svmrc_cm <- confusionMatrix(factor(svmrc_predicted),factor(dat_test$DEFAULT))

result_other <- rbind(result_other, with(svmrc_cm, 
                                         c(Method = "svmRadialCost",
                                           Accuracy = overall["Accuracy"],
                                           Balanced = byClass["Balanced Accuracy"],
                                           Sensitivity = byClass["Sensitivity"],
                                           Specificity = byClass["Specificity"],
                                           F1 = byClass["F1"])))
result_other %>% knitr::kable(caption="PREDICTION RESULT", digits=4)


#---------------------------- . ----------------------------#
#    PREDICT USING svmRadialSigma
#---------------------------- . ----------------------------#
svmrs_fit <- train(data=dat_train_simple, DEFAULT~.,
                   method = "svmRadialSigma")
svmrs_predicted <- predict(svmrs_fit, dat_test)
svmrs_cm <- confusionMatrix(factor(svmrs_predicted),factor(dat_test$DEFAULT))

result_other <- rbind(result_other, with(svmrs_cm, 
                                         c(Method = "svmRadialSigma",
                                           Accuracy = overall["Accuracy"],
                                           Balanced = byClass["Balanced Accuracy"],
                                           Sensitivity = byClass["Sensitivity"],
                                           Specificity = byClass["Specificity"],
                                           F1 = byClass["F1"])))
result_other %>% knitr::kable(caption="PREDICTION RESULT", digits=4)


sink()
