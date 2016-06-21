"""
Programa teste para entender os P-a-P dos metodos utilizados no
algoritmo de FoF.

Author: Patricia NOVAIS
Maio de 2016
"""
import pandas as pd

class FoF():
    def __init__(self):
        self.df_fof = ""
        self.dist = ""
        self.fnames = ""


    def read_columns(self,names):
        self.fnames = names
        self.posicao = pd.read_table(self.fnames, header=None, delim_whitespace=True)
        self.posicao.columns = ['x','y']
        return self.posicao

    def mkClusters(self):
        """
        create the list of clusters using the friends of friends def.
        """
        ## create auxiliary variable to keep track of checked galaxies
        ## 0 -> not checked yet
        self.checked = self.nTot*[0]
        ## create the list which will store the clusters
        self.clusters = []
        ## find clusters for each gal in pos
        for gal in self.galaxies:
            if(self.checked[gal] == 0):
                self.checkOne(gal)


ff = FoF()
pos = ff.read_columns('fof.txt')
ff.df_fof = 'teste'
ff.dist = 1.0

print(pos)
