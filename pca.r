setwd('/home/pnovais/Dropbox/DOUTORADO/JPLUS_clusters/gal_A2589A2593/')
data1 = read.table("contagens.txt")
colnames(data1) <- c("x","y",'u','g','r','i','z','J0378','J0395','J0410','J0430','J0515','J0660','J861')
pairs(data1)
cont.pca1 <- princomp(data1,scores = TRUE, cor=TRUE)
summary(cont.pca1)
plot(cont.pca1)
biplot(cont.pca1)
cont.pca1$loadings
cont.pca1$scores
data2 <- cbind(data1$u, data1$g, data1$r, data1$i, data1$z, data1$J0378, data1$J0395,data1$J0410, data1$J0430,data1$J0515,data1$J0660,data1$J861)

data3 <- cbind(data1$u, data1$g, data1$r, data1$i, data1$z)
colnames(data3) <- c('u','g','r','i','z')
pairs(data3)
cont.pca3 <- princomp(data3,scores = TRUE, cor=TRUE)
summary(cont.pca3)
plot(cont.pca3)
biplot(cont.pca3)