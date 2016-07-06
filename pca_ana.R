# PCA in r
# What method 2 does ?
ptm <- proc.time()
DF <- data.frame(read.table("gal_old.txt"))
# Matrix normalizada
Dmean <- as.matrix(DF)
for (i in 1:length(DF[1,])) {
  Dmean[,i]=(Dmean[,i]-mean(DF[,i]))
}
# Matrix covariancia
Dcov <- cov(Dmean)
# Autovetores & Autovalores
Deigen <- data.frame(eigen(Dcov)$vectors)
# PC matrix
DPC <- Dmean-Dmean
for (i in 1:length(DF[,1])) { #linha
  for (j in 1:length(DF[1,])) { #coluna
    DPC[i,j]=sum(Dmean[i,] * Deigen[j,])
  }
}
# Data out
Dout <- function(data,fileout) {
          mylines <- as.data.frame(data)
          mylines <- noquote(prettyNum(round(mylines,3), nsmall = 3))
          write.table(mylines,fileout,sep=" ",quote=F,row.names=F)
          return()
          }
Dout(Deigen,"PC.txt")
proc.time() - ptm
# Method 2 ---> fast
ptm <- proc.time()
DF <- data.frame(read.table("gal_old.txt", nrows=1000))
Deigen <- prcomp(DF)$rotation
# Data out
Dout <- function(data,fileout) {
          mylines <- as.data.frame(data)
          mylines <- noquote(prettyNum(round(mylines,16), nsmall = 16))
          write.table(mylines,fileout,sep=" ",quote=F,row.names=F)
          return()
          }
Dout(Deigen,"PC.txt")
proc.time() - ptm
print('...end...')

