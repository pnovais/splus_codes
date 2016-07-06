# > source("meu_pca.R")
# leitura do arquivo de dados

mydata <- read.table(file="gal_old.txt", header=FALSE)

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
