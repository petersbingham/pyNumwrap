# -*- coding: utf-8 -*-
import inspect
from six.moves import builtins
import cmath
try:
    import mpmath
except:
    pass
try:
    import numpy as np
except:
    pass
try:
    import sympy as sy
except:
    pass

##########################################################
####################### Constants ########################
##########################################################

mode_python = 0
mode_mpmath = 1
dps_default_python = 25
dps_default_mpmath = 100

##########################################################
####################### Variables ########################
##########################################################

mode = mode_python
dps = dps_default_python
pi = cmath.pi

##########################################################
################# Configuration Functions ################
##########################################################

def usePythonTypes(dpsNew=dps_default_python):
    global mode, dps, pi
    mode = mode_python
    dps = dpsNew
    pi = cmath.pi

def useMpmathTypes(dpsNew=dps_default_mpmath):
    global mode, dps, pi
    mode = mode_mpmath
    dps = dpsNew
    pi = mpmath.pi
    mpmath.mp.dps = dps

############### BASIC TYPES ###############

# For convenience:
class mpf(mpmath.mpf):
    pass

# np.percentile need these overrides.
class mpc(mpmath.mpc):
    def __lt__(self, other):
        return self.real < other.real
        
    def __le__(self, other):
        return self.real <= other.real
        
    def __gt__(self, other):
        return self.real > other.real
    
    def __ge__(self, other):
        return self.real >= other.real

def float(val):
    if mode == mode_python:
        return builtins.float(val)
    else:
        return mpmath.mpf(val)

def complex(val):
    if mode == mode_python:
        return builtins.complex(val)
    else:
        if type(val) is str or type(val) is unicode:
            if 'nan' in val:
                return mpmath.mpc(real='nan',imag='nan')
            real = None
            imag = None
            delim = None
            if '+' in val[1:]:
                delim = '+'
            elif '-' in val[1:]:
                delim = '-'
            if delim is None:
                if 'j' in val:
                    imag = val.replace('j','')
                else:
                    real = val
            else:
                index = val[1:].find(delim) + 1
                real = val[:index]
                imag = val[index:].replace('j','')
            return mpmath.mpc(real=real,imag=imag)
        else:
            return mpmath.mpc(val)

############### SYMPY CONVERSIONS ###############

def toSympy(val):
    if mode == mode_python:
        return val
    else:
        return sy.Float(str(val.real),dps) + sy.Float(str(val.imag),dps)*sy.I

def fromSympy(val):
    if mode == mode_python:
        return complex(val)
    else:
        a = sy.simplify(val)
        return mpmath.mpmathify(a)

def toSympyMatrix(mat):
    if mode == mode_python:
        return sy.Matrix(mat)
    else:
        symMat = sy.zeros(mat.rows, mat.cols)
        for r in range(mat.rows):
            for c in range(mat.cols):
                symMat[r,c] = mat[r,c]
        return symMat

def fromSympyMatrix(mat):
    newMat = zeroMatrix(mat.shape[0], mat.shape[1])
    for r in range(mat.shape[0]):
        for c in range(mat.shape[0]):
            newMat[r,c] = fromSympy(mat[r,c])
    return newMat

############### BASIC OPERATIONS ###############

def percentile(a, q, axis=None, out=None, overwrite_input=False,
               interpolation='linear', keepdims=False):
    # Currently don't support percentile for mp types. Just convert the type.
    if mode == mode_python:
        return np.percentile(a, q, axis, out, overwrite_input, interpolation,
                             keepdims)
    else:
        return np.percentile(map(lambda v: mpc(v), a), q, axis, out, 
                             overwrite_input, interpolation, keepdims)

def pow(x, y):
    if mode == mode_python:
        return builtins.pow(x, y)
    else:
        return mpmath.power(x, y)

def exp(x):
    if mode == mode_python:
        return cmath.exp(x)
    else:
        return mpmath.exp(x)

def sqrt(x):
    if mode == mode_python:
        return cmath.sqrt(x)
    else:
        return mpmath.sqrt(x)

def tan(x):
    if mode == mode_python:
        return cmath.tan(x)
    else:
        return mpmath.tan(x)

def polar(x):
    if mode == mode_python:
        return cmath.polar(x)
    else:
        return mpmath.polar(x)

def rootsSym(symPoly, **kwargs):
    if mode == mode_python:
        coeffs = symPoly.all_coeffs()
        mappedCoeffs = map(lambda val: complex(val), coeffs)
        return np.roots(mappedCoeffs, **kwargs)
    else:
        return symPoly.nroots(**kwargs)   

############### MATRIX TYPES ###############

def matrix(val):
    if mode == mode_python:
        return np.matrix(val, dtype=np.complex128)
    else:
        return mpmath.matrix(val)

def zeroMatrix(rows, cols=None):
    if cols is None:
        cols = rows
    if mode == mode_python:
        return np.matrix(np.zeros((rows, cols), dtype=np.complex128))
    else:
        return mpmath.zeros(rows, cols)

def identity(sz):
    if mode == mode_python:
        return np.matrix(np.identity(sz, dtype=np.complex128))
    else:
        return mpmath.eye(sz)

############# MATRIX CHARACTERISTICS #############
    
def shape(mat):
    if mode == mode_python:
        return mat.shape
    else:
        return (mat.rows, mat.cols)

def size(mat):
    if mode == mode_python:
        return mat.size
    else:
        return mat.rows*mat.cols

def isSquare(mat):
    shp = shape(mat)
    return shp[0] == shp[1]

def isIdentity(mat, rtol=1e-05, atol=1e-08):
    if not isSquare(mat):
        return False
    iMat = identity(shape(mat)[0])
    return areMatricesClose(iMat, mat, rtol, atol)

def isUnitary(mat, rtol=1e-05, atol=1e-08):
    if not isSquare(mat):
        return False
    iMat = identity(shape(mat)[0])
    tcMat = transpose(conjugate(mat))
    return areMatricesClose(iMat, mat*tcMat, rtol, atol)

############### MATRIX OPERATIONS ###############

def absolute(mat):
    if mode == mode_python:
        return np.absolute(mat)
    else:
        absMat = mpmath.matrix(mat.rows, mat.cols)
        for i in range(mat.rows):
            for j in range(mat.cols):
                absMat[i,j] = abs(mat[i,j])
        return absMat

def transpose(mat):
    if mode == mode_python:
        return np.transpose(mat)
    else:
        return mat.T

def conjugate(mat):
    if mode == mode_python:
        return np.conjugate(mat)
    else:
        return mat.conjugate()

def invert(mat):
    if mode == mode_python:
        return np.linalg.inv(mat)
    else:
        return mpmath.inverse(mat)

def dot(matA, matB):
    if mode == mode_python:
        return np.dot(matA, matB)
    else:
        return matA * matB

def unitaryOp(mat):
    return transpose(conjugate(mat))

def getRow(mat, m):
    if mode == mode_python:
        return mat[m].tolist()[0]
    else:
        row = []
        for n in range(mat.cols):
            row.append(mat[m,n])
        return row

def getCol(mat, n):
    if mode == mode_python:
        return mat[:,n].tolist()[0]
    else:
        col = []
        for m in range(mat.rows):
            row.append(mat[m,n])
        return col

def getVector(mat, i, isCol=False):
    if not isCol:
        vec = getRow(mat, i)
    else:
        vec = getCol(mat, i)
    if mode == mode_python:
        return np.array(vec)
    else:
        return mpmath.matrix(vec)

def copyRow(src_mat, dest_mat, m):
    newMat = dest_mat.copy()
    if mode == mode_python:
        for n in range(newMat.shape[1]):
            newMat[m,n] = src_mat[m,n]
    else:
        for n in range(newMat.cols):
            newMat[m,n] = src_mat[m,n]
    return newMat

def det(mat):
    if mode == mode_python:
        return np.linalg.det(mat)
    else:
        return mpmath.det(mat)

def sumElements(mat):
    if mode == mode_python:
        XS = 0.0
        for x in np.nditer(mat, flags=['refs_ok']):
            XS += x
    else:
        XS = mpmath.mpc(0.0)
        for i in range(mat.rows):
            for j in range(mat.cols):
                XS += mat[i,j]
    return XS

def trace(mat):
    if mode == mode_python:
        return np.trace(mat)
    else:
        t = mpmath.mpc(0.0)
        for i in range(mat.rows):
            t += mat[i,i]
        return t

def atanElements(mat):
    if mode == mode_python:
        return np.arctan(mat)
    else:
        at = mpmath.matrix(mat.rows, mat.cols)
        for i in range(mat.rows):
            for j in range(mat.cols):
                at[i,j] = mpmath.atan(mat[i,j])
        return at

def adjugate(mat):
    symMat = toSympyMatrix(mat)
    return fromSympyMatrix(symMat.adjugate())

def lin_solve(mat, vec, **kwargs):
    if mode == mode_python:
        return np.linalg.solve(mat, vec, **kwargs)
    else:
        return mpmath.qr_solve(mat, vec, **kwargs)[0]

def diagonalise(mat):
    if mode == mode_python:
        w, v = np.linalg.eig(mat)
        P = np.transpose(np.matrix(v, dtype=np.complex128))
        return np.dot(P, np.dot(mat, np.linalg.inv(P)))
    else:
        w, v = mpmath.eig(mat)
        P = mpmath.matrix(v).T
        return P * mat * P**-1

############### MATRIX COMPARISONS ###############

def areMatricesClose(mat1, mat2, rtol=1e-05, atol=1e-08, equal_nan=False):
    if mode == mode_python:
        return np.allclose(mat1, mat2, rtol, atol, equal_nan)
    else:
        if shape(mat1) != shape(mat2):
            return False
        for r in range(mat1.rows):
            for c in range(mat1.cols):
                a = mat1[r,c]
                b = mat2[r,c]
                if mpmath.isnan(a) and mpmath.isnan(b) and equal_nan:
                    pass
                elif not numCmp(a, b, atol, rtol):
                    return False
    return True

############### OTHER ###############

def formattedFloatString(val, dps):
    if mode == mode_python:
        return ("{:1."+str(dps)+"f}").format(val)
    else:
        return mpmath.nstr(val, mpIntDigits(val)+dps)

def formattedComplexString(val, dps):
    if val.imag < 0.0:
        signStr = ""
    else:
        signStr = "+"
    rstr = formattedFloatString(val.real, dps)
    istr = formattedFloatString(val.imag, dps)+"j"
    return rstr + signStr + istr

def floatList(lst):
    return str(map(lambda x:str(x),lst)).replace("'","")

def mpIntDigits(num):
    if not mpmath.almosteq(num,0):
        a = mpmath.log(abs(num), b=10)
        b = mpmath.nint(a)
        if mpmath.almosteq(a,b):
            return int(b)+1
        else:
            c = mpmath.ceil(a)
            try:
                return int(c)
            except:
                pass
    else:
        return 0

def numCmp(a, b, atol, rtol):
    return abs(a-b) <= atol + rtol * abs(b)

def getArgDesc(func, args, ignore=None):
    a = inspect.getargspec(func)
    if a.defaults is not None:
        d = dict(zip(a.args[-len(a.defaults):],a.defaults))
    else:
        d = {}
    d.update(args)
    argStr = "("
    first = True
    for arg in a.args:
        if arg in d:
            if ignore is None or arg not in ignore:
                if not first:
                    argStr += ","
                else:
                    first = False
                argStr += arg + " " + str(d[arg])
    argStr += ")"
    return argStr
