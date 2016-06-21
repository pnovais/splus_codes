#!/usr/bin/python2.3

"""
   ff.py: simple friends of friends algorithm

   this module is meant to demonstrate the top down approach
   for program design for educational purposes.

   Mon Jan 10 21:25:03 HST 2005

   (c) Istvan Szapudi, Institute for Astronomy, University of Hawaii
"""

class FriendsOfFriends:
    """
    from a list of object (galaxy) coordinates in n-dimension
    (n=1,2,3) create the list of clusters defined by friends of friends

    """

    def __init__(self,fname,length):

        """
        file fname contains ascii positions (1-3 dimensions)
        x [y z]

        with comments allowed starting with

        #....comment

        input is stored in pos, a list of galaxy positions

        [[x,y,z],...]

        A f.o.f. cluster is defined s.t. each member has another
        galaxy within distance length.

        creates a list of the clusters in self.clusters

        [[galaxy1,galaxy2,...,galaxyN],....]

        """
        self.length=length
        self.fname=fname
        ## read galaxy positions into a list
        self.pos = self.readColumns(fname)
        ## record to total number of objects and the dimension
        self.nTot = len(self.pos)
        self.dim = len(self.pos[0])
        # this is only for clarity: galaxies=indexes
        self.galaxies = range(self.nTot)
        ## create the list of clusters
        self.mkClusters()

    def readColumns(self,infile):
        """
        read an ascii list of x1 [x2 ...] from an ascii file
        with # allowed for comments
        """
        pos = []
        for line in file(infile):
             if line[0]=='#':
                 continue
             row = [float(val) for val in line.split()]
             pos.append(row)
        return pos


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


    def checkOne(self,gal):
        """
        check gal (a galaxy represented by its index) for cluster membership
        return the cluster defined by this galaxy
        """
        ## all galaxies that belong to previous clusters are already checked
        ## therefore create an empty cluster
        newCluster = []
        ## the galaxy itself will be a member of its own cluster
        newCluster.append(gal)
        self.checked[gal] = 1
        nCluster = 1 ## we store the number of objects in the cluster
        ## start with first element of the cluster
        k = 0
        while k < nCluster:
            ## add all the Friends of galaxy newCluster[k] to the cluster
            ## return the number of objects added
            nCluster += self.addFriends(newCluster,k)
            k += 1

        ## finally append the newCluster in the
        self.clusters.append(newCluster)


    def addFriends(self,cluster,k):
        """
           add all galaxies to cluster which are closer (or equal)
           then self.length
           (i.e.Friends) of the k-th cluster member

           return the total number of objects added
        """
        ## this will store the number objects added
        n = 0
        for gal in self.galaxies:
            if(self.checked[gal] == 0):
                if(self.calcDist(gal,k) <= self.length):
                    ## if friend we add to the cluster
                    cluster.append(gal)
                    ## record the number added
                    n += 1
                    ## this is checked
                    self.checked[gal] = 1

        return n


    def calcDist(self,gal1,gal2):
        """
        return the Euclidean distance between galaxies gal1 and gal2
        """
        from math import sqrt as sqrt

        d = 0.0;
        for i in range(self.dim):
            d += (self.pos[gal1][i]-self.pos[gal2][i])**2
        return sqrt(d)


    def numberOfClusters(self, n0=0):
        """
        returns the number of clusters greater or equal to n0
        """
        nt = len(self.clusters)
        nc = 0
        for i in range(nt):
            if len(self.clusters[i]) >= n0:
                   nc += 1
        return nc

def test():
    """
    unit test for ff.py
    """
    ff=FriendsOfFriends('ff.dat',0.5)
##    print ff.pos
    print ff.clusters
    print ff.numberOfClusters(2)


def test2D():
    """
    unit test for ff.py
    """
    ff=FriendsOfFriends('ff2d.dat',0.5)
##    print ff.pos
    print ff.clusters
    print ff.numberOfClusters(2)

if __name__=='__main__':
    test()
    test2D()

## [szapudi@joshu lecture1]$ ./ff.py
## [[0, 1], [2, 3, 4], [5], [6]]
## 2
## [[0, 1], [2, 3, 4], [5], [6]]
## 2
