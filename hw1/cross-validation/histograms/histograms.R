# CS 31 HW1
# Histograms for each feature

# load and separate data
dta1 = read.csv(file="~/PycharmProjects/cs74/HW1/data/hw1_trainingset.csv")
dta2 = read.csv(file="~/PycharmProjects/cs74/HW1/data/hw1_testset.csv")

# separate out features
f1_1 = dta1[,1]
f2_1 = dta1[,2]
f3_1 = dta1[,3]
f4_1 = dta1[,4]
f5_1 = dta1[,5]
f6_1 = dta1[,6]

f1_2 = dta2[,1]
f2_2 = dta2[,2]
f3_2 = dta2[,3]
f4_2 = dta2[,4]
f5_2 = dta2[,5]
f6_2 = dta2[,6]

# graph
hist(x=f1_1)
hist(x=f1_2)

hist(x=f2_1)
hist(x=f2_2)

hist(x=f3_1)
hist(x=f3_2)

hist(x=f4_1)
hist(x=f4_2)

hist(x=f5_1)
hist(x=f5_2)

hist(x=f6_1)
hist(x=f6_2)