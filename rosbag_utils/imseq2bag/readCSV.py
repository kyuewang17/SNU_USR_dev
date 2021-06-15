import bpy
import os


class PointCloud():
    def __init__(self,fname):
        self.fname=fname
        self.header=[]
        self.data=[]
        self.points=[]
        self.headCols={}
        self.X=None
        self.Y=None
        self.Z=None

    def loadPoints(self,fname=None, delim=','):
        """loadPoints(fname=None)
           This is very dumb at the moment.  It simply reads the
           file from disk assuming its a proper .csv file and stores
           it internally without error checking.
           If fname is None and self.fname is none, returns False
           If fname is set, then assigns new filepath to self.fname
        """

        if fname: self.fname=fname
        if not self.fname: return False

        fid=open(self.fname,'r')
        self.header=[v.strip() for v in fid.readline().split(delim)]

        self.headCols={}
        for col,h in enumerate(self.header):
            self.headCols[h]=col

        self.data=[]
        for line in fid.readlines():
            self.data.append([float(v.strip()) for v in line.split(delim)])

        fid.close()

    def getHeader(self):
        """getHeader()
           Returns the header.  Seems silly to have an accessor method
           in python, but I may add some bells and whistles in the future,
           so I recommend using using this function just in case.
        """
        return header

    def assignPoints(self,X=None,Y=None,Z=None):
        """assignPoints(X=None,Y=None,Z=None)
           Assigns specific columns of the data to be X, Y and Z points
           in the point cloud.
           If no column names are give, then the point array is cleared.
        """

        # Clearning out the array no matter what
        self.points=[]

        # Not the nicest thing to do, but if the function isn't passed a
        # valid header, I silently treat it as a None
        if X not in self.header: X=None
        if Y not in self.header: Y=None
        if Z not in self.header: Z=None

        self.X = X
        self.Y = Y
        self.Z = Z

        # Am I being given everything I need to do something, or do I
        # clear the list of points?
        ndim=3
        if X is None: ndim = ndim-1
        if Y is None: ndim = ndim-1
        if Z is None: ndim = ndim-1

        if ndim == 0:
            return 0

        # Get the header positions
        zeroArray=[0]*len(self.data)
        if X is None: xp=-1
        else: xp=self.header.index(X)
        if Y is None: yp=-1
        else: yp=self.header.index(Y)
        if Z is None: zp=-1
        else: zp=self.header.index(Z)

        xx=[v[xp] if xp>=0 else 0 for v in self.data]
        yy=[v[yp] if yp>=0 else 0 for v in self.data]
        zz=[v[zp] if zp>=0 else 0 for v in self.data]

        self.points=[v for v in zip(xx,yy,zz)]

        return len(self.points)