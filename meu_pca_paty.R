# > source("meu_pca.R")
# leitura do arquivo de dados

data <- read.table(file="gal_old.txt", header=FALSE)
mydata <- data[3:14]
mydata_coord <- data[1:2]

# pca com variaveis com media zero e variancia unitaria

mydata.pca <- prcomp(mydata, retx=TRUE, center=TRUE, scale=TRUE)

# raiz quadrada dos autovalores (singular value)

sd <- mydata.pca$sdev

# autovalores

lambda <- sd^2

# autovetores

autovec <- mydata.pca$rotation

# nomes das colunas

rownames(autovec)

# componentes principais

pc <- mydata.pca$x

# preparacao para impressao

nl <- min(length(lambda),10)
na <- min(length(autovec)^0.5, 10)

# salva os dados num arquivo

write.table(lambda[1:nl],"Rpca.res")
write.table(autovec[1:na,1:na],"Rpca.res", append = TRUE)
write.table(pc[,1:na],"Rpca.res", append = TRUE)

# distance biplot
pdf("Rpca.pdf")
plot(pc[,1], pc[,2], xlab="PCA 1", ylab="PCA 2",type="n", xlim=c(min(pc[,1:2]), max(pc[,1:2])),ylim=c(min(pc[,1:2]), max(pc[,1:2])))
points(pc[,1], pc[,2])
dev.off()

# 

pc
teste <- cbind(mydata_coord,mydata,pc)

#join(mydata_coord,data,pc)
plot(teste$V3-teste$V5,teste$PC1)
plot(teste$V3,teste$PC1)
plot(teste$V4-teste$V5,teste$PC1)
plot(teste$V6-teste$V5,teste$PC1)
plot(teste$V7-teste$V5,teste$PC1)
plot(teste$V8-teste$V5,teste$PC1)
plot(teste$V9-teste$V5,teste$PC1)
plot(teste$V10-teste$V5,teste$PC1)
plot(teste$V11-teste$V5,teste$PC1)
plot(teste$V12-teste$V5,teste$PC1)
plot(teste$V13-teste$V5,teste$PC1)
plot(teste$V14-teste$V5,teste$PC1)
