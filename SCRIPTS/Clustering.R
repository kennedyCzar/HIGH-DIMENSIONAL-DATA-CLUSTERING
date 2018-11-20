library(tidyverse)
library(cluster)
library(factoextra)

#set the working directory
setwd('D:\\GIT PROJECT\\CLUSTERING\\datasets')

dataset = read.csv('df_Peos.csv', sep = ',', header = TRUE)

#device<- dataset[, 2]; device
row.names(dataset)<- dataset[, 2]
df<- dataset[, 3:254]


#scale the dataset
df<- scale(df)

df<- df[, 1:250]

df<- data.frame(df)
df$X5comp.1<- NULL
df$X5comp.2<- NULL
df$X5comp<- NULL
#get the distance function
distance<- get_dist(df)
fviz_dist(distance, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))


#set seed
set.seed(123)
#perform kmeans on the dataset
k2<- kmeans(df, centers = 2, nstart = 25); k2

fviz_cluster(k2, data = df)


#finding the optimum clusters
fviz_nbclust(df, kmeans, method = 'silhouette')


#multiple clustering results
k3 <- kmeans(df, centers = 3, nstart = 25)
k4 <- kmeans(df, centers = 4, nstart = 25)
k5 <- kmeans(df, centers = 5, nstart = 25)

# plots to compare
p1 <- fviz_cluster(k2, geom = "point", data = df) + ggtitle("k = 2")
p2 <- fviz_cluster(k3, geom = "point",  data = df) + ggtitle("k = 3")
p3 <- fviz_cluster(k4, geom = "point",  data = df) + ggtitle("k = 4")
p4 <- fviz_cluster(k5, geom = "point",  data = df) + ggtitle("k = 5")

library(gridExtra)
grid.arrange(p1, p2, p3, p4, nrow = 2)

####
#final cluster using 4clusters
kfinal<- kmeans(df, centers = 4, nstart = 25); kfinal

fviz_cluster(kfinal, data = df) + ggtitle("Sheet df_BR: Clustered data with respect to Device name containing 4clusters")


