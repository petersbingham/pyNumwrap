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
    import sympy as sym
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
typeLocked = False

##########################################################
################# Configuration Functions ################
##########################################################

def use_python_types(dpsNew=dps_default_python):
    global mode, dps, pi, typeLocked
    if typeLocked:
        raise Exception("Type locked")
    mode = mode_python
    dps = dpsNew
    pi = cmath.pi

def use_mpmath_types(dpsNew=dps_default_mpmath):
    global mode, dps, pi, typeLocked
    if typeLocked:
        raise Exception("Type locked")
    mode = mode_mpmath
    dps = dpsNew
    pi = mpmath.pi
    mpmath.mp.dps = dps

def set_type_mode(mode, dps=None):
    if mode is None or mode != mode_mpmath:
        if dps is None:
            use_python_types(dps_default_python)
        else:
            use_python_types(dps)
    else:
        if dps is None:
            use_mpmath_types(dps_default_mpmath)
        else:
            use_mpmath_types(dps)

def lockType():
    global typeLocked
    typeLocked = True

def getConfigString():
    if mode == mode_python:
        return "numpy"
    else:
        return "mpmath_"+str(dps)

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
        return mpmath.mpf(mpmath.mpmathify(val))

def complex(val):
    if mode == mode_python:
        return builtins.complex(val)
    else:
        return mpmath.mpc(mpmath.mpmathify(val))

############### SYMPY CONVERSIONS ###############

def to_sympy(val):
    if mode == mode_python:
        return val
    else:
        return sym.Float(str(val.real),dps) + sym.Float(str(val.imag),dps)*sym.I

def from_sympy(val):
    sval = sym.simplify(val)
    if mode == mode_python:
        return complex(sval)
    else:
        try:
            return complex(sval)
        except TypeError:
            # Exception in earlier versions of sympy (and possibly mpmath).
            # See https://stackoverflow.com/questions/49211887/convert-a-sympy-poly-with-imaginary-powers-to-an-mpmath-mpc
            comps = sval.subs('I',1.0j)
            real = sym.re(comps)
            imag = sym.im(comps)
            return mpmath.mpc(real,imag)

def to_sympy_matrix(mat):
    if mode == mode_python:
        return sym.Matrix(mat)
    else:
        sym_mat = sym.zeros(mat.rows, mat.cols)
        for r in range(mat.rows):
            for c in range(mat.cols):
                symMat[r,c] = mat[r,c]
        return sym_mat

def from_sympy_matrix(mat):
    new_mat = zero_matrix(mat.shape[0], mat.shape[1])
    for r in range(mat.shape[0]):
        for c in range(mat.shape[0]):
            new_mat[r,c] = from_sympy(mat[r,c])
    return new_mat

############### BASIC OPERATIONS ###############

def _app_to_mpmath(x, fun):
    try:
        len(x) # Determine if simple type or matrix.
        return apply_fun_to_elements(x, lambda i,j,x:fun(x))
    except TypeError:
        return fun(x)

def abs(x):
    if mode == mode_python:
        return builtins.abs(x)
    else:
        return mpmath.fabs(x)

def pow(x, y):
    if mode == mode_python:
        return builtins.pow(x, y)
    else:
        return mpmath.power(x, y)

def exp(x):
    if mode == mode_python:
        return np.exp(x)
    else:
        return _app_to_mpmath(x, mpmath.exp)

def sqrt(x):
    if mode == mode_python:
        return np.lib.scimath.sqrt(x) # This version OK with neg nums.
    else:
        return _app_to_mpmath(x, mpmath.sqrt)

def tan(x):
    if mode == mode_python:
        return np.tan(x)
    else:
        return _app_to_mpmath(x, mpmath.tan)

def arctan(x):
    if mode == mode_python:
        return np.arctan(x)
    else:
        return _app_to_mpmath(x, mpmath.atan)

def log(x):
    if mode == mode_python:
        return np.log(x)
    else:
        return _app_to_mpmath(x, mpmath.log)

# Currently does not support matrix types. Prob is that np.matrices cannot
# the hold sequences that polar returns.
def polar(x):
    if mode == mode_python:
        return cmath.polar(x)
    else:
        return mpmath.polar(x)

def roots_sym(symPoly, **kwargs):
    if mode == mode_python:
        coeffs = symPoly.all_coeffs()
        mapped_coeffs = map(lambda val: from_sympy(val), coeffs)
        return np.roots(mapped_coeffs)
    else:
        if "symPoly_nroots" in kwargs:
            roots = symPoly.nroots(**kwargs["symPoly_nroots"])
        else:
            roots = symPoly.nroots()
        return map(lambda val: from_sympy(val), roots)

def percentile(a, q, axis=None, out=None, overwrite_input=False,
               interpolation='linear', keepdims=False):
    # Currently don't support percentile for mp types. Just convert the type.
    if mode == mode_python:
        return np.percentile(a, q, axis, out, overwrite_input, interpolation,
                             keepdims)
    else:
        return np.percentile(map(lambda v: mpc(v), a), q, axis, out, 
                             overwrite_input, interpolation, keepdims)

def gradient(f, varargs):
    reals = map(lambda x:x.real, f)
    imags = map(lambda x:x.imag, f)
    real_grads = np.gradient(reals, varargs)
    imag_grads = np.gradient(imags, varargs)
    return [real_grads[i]+imag_grads[i]*1.j for i in range(len(real_grads))]

############### MATRIX TYPES ###############

def matrix(val):
    if mode == mode_python:
        return np.matrix(val, dtype=np.complex128)
    else:
        return mpmath.matrix(val)

def vector(val):
    if mode == mode_python:
        return np.array(val)
    else:
        return mpmath.matrix(val)

def zero_matrix(rows, cols=None):
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
        shp = mat.shape
        if len(shp) == 1:
            return (1, shp[0])
        return shp
    else:
        return (mat.rows, mat.cols)

def size(mat):
    if mode == mode_python:
        return mat.size
    else:
        return mat.rows*mat.cols

def is_square(mat):
    shp = shape(mat)
    return shp[0] == shp[1]

def is_identity(mat, rtol=1e-05, atol=1e-08):
    if not is_square(mat):
        return False
    i_mat = identity(shape(mat)[0])
    return are_matrices_close(i_mat, mat, rtol, atol)

def is_unitary(mat, rtol=1e-05, atol=1e-08):
    if not is_square(mat):
        return False
    i_mat = identity(shape(mat)[0])
    return are_matrices_close(i_mat, unitary_op(mat), rtol, atol)

############### MATRIX OPERATIONS ###############

def absolute(mat):
    if mode == mode_python:
        return np.absolute(mat)
    else:
        abs_mat = mpmath.matrix(mat.rows, mat.cols)
        for i in range(mat.rows):
            for j in range(mat.cols):
                abs_mat[i,j] = abs(mat[i,j])
        return abs_mat

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

def dagger(mat):
    return transpose(conjugate(mat))

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

def unitary_op(mat):
    return transpose(conjugate(mat)) * mat

def get_row(mat, i):
    if mode == mode_python:
        return np.array(mat[i].tolist()[0])
    else:
        row = []
        for j in range(mat.cols):
            row.append(mat[i,j])
        return mpmath.matrix([row])

def get_col(mat, j):
    if mode == mode_python:
        return np.array(mat[:,j].tolist())
    else:
        col = []
        for i in range(mat.rows):
            col.append(mat[i,j])
        return mpmath.matrix(col)

def get_diag(mat):
    if mode == mode_python:
        return mat.diagonal()
    else:
        diag = []
        for m in range(mat.rows):
            diag.append(mat[m,m])
        return mpmath.matrix([diag])

def get_vector(mat, i, col=False):
    if not col:
        return get_row(mat, i)
    else:
        return get_col(mat, i)

def copy_row(src_mat, dest_mat, m):
    new_mat = dest_mat.copy()
    if mode == mode_python:
        for n in range(new_mat.shape[1]):
            new_mat[m,n] = src_mat[m,n]
    else:
        for n in range(new_mat.cols):
            new_mat[m,n] = src_mat[m,n]
    return new_mat

def det(mat):
    if mode == mode_python:
        return np.linalg.det(mat)
    else:
        return mpmath.det(mat)

def sum_elements(mat):
    if mode == mode_python:
        sum = 0.0
        for x in np.nditer(mat, flags=['refs_ok']):
            sum += x
    else:
        sum = mpmath.mpc(0.0)
        for i in range(mat.rows):
            for j in range(mat.cols):
                sum += mat[i,j]
    return sum

def apply_fun_to_elements(mat, fun_ref):
    new_mat = mat.copy()
    if mode == mode_python:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                new_mat[i,j] = fun_ref(i, j, mat[i,j])
    else:
        sum = mpmath.mpc(0.0)
        for i in range(mat.rows):
            for j in range(mat.cols):
                new_mat[i,j] = fun_ref(i, j, mat[i,j])
    return new_mat

def trace(mat):
    if mode == mode_python:
        return np.trace(mat)
    else:
        t = mpmath.mpc(0.0)
        for i in range(mat.rows):
            t += mat[i,i]
        return t

def atan_elements(mat):
    if mode == mode_python:
        return np.arctan(mat)
    else:
        at = mpmath.matrix(mat.rows, mat.cols)
        for i in range(mat.rows):
            for j in range(mat.cols):
                at[i,j] = mpmath.atan(mat[i,j])
        return at

def adjugate(mat):
    sym_mat = to_sympy_matrix(mat)
    return from_sympy_matrix(sym_mat.adjugate())

def lin_solve(mat, vec):
    if mode == mode_python:
        return np.linalg.solve(mat, vec)
    else:
        return mpmath.qr_solve(mat, vec)[0]

def lin_solve_homo(mat):
    if mode == mode_python:
        _,v,_ = np.linalg.svd(mat)
        return transpose(np.asmatrix(v))
    else:
        raise NotImplementedError

def diagonalise(mat):
    if mode == mode_python:
        _, v = np.linalg.eig(mat)
        P = np.transpose(np.matrix(v, dtype=np.complex128))
        return np.dot(P, np.dot(mat, np.linalg.inv(P)))
    else:
        _, v = mpmath.eig(mat)
        return v**-1 * mat * v

def eigenvalues(mat, sort=True):
    if mode == mode_python:
        e, _ = np.linalg.eig(mat)
        if sort:
            idx = np.argsort(e)
            return e[idx]
        else:
            return e
    else:
        e, v = mpmath.eig(mat)
        if sort:
            e, v = mpmath.eig_sort(e, v)
        return mpmath.matrix(e)

############### MATRIX COMPARISONS ###############

def are_matrices_close(mat1, mat2, rtol=1e-05, atol=1e-08, equal_nan=False):
    if mode == mode_python:
        return np.allclose(mat1, mat2, rtol, atol, equal_nan)
    else:
        # Just make sure we have matrices and not lists. np.allclose supports
        # lists. List support is useful for tests as can have test data that is
        # common to both representations in list form.
        mat1_ = matrix(mat1)
        mat2_ = matrix(mat2)
        if shape(mat1_) != shape(mat2_):
            return False
        for r in range(mat1_.rows):
            for c in range(mat1_.cols):
                a = mat1_[r,c]
                b = mat2_[r,c]
                if mpmath.isnan(a) and mpmath.isnan(b) and equal_nan:
                    pass
                elif not num_cmp(a, b, atol, rtol):
                    return False
    return True

############### OTHER ###############

def _check_ztol(num, num_str, ztol):
    if ztol:
        if abs(num) < ztol:
            num_str = "0"
    return num_str

# See mpmath doc for kwargs description. Appears to be an mpmath bug where
# kwargs do not apply correctly to real and imag component, so split here.
# To always use floating-point format (eg 1.0 -> 1.0e+0) use kwargs:
# min_fixed >= max_fixed. Always fixed min_fixed = -inf and max_fixed = +inf
def num_str_pair(val, sig_digits=15, strip_zeros=False, min_fixed=-3,
                 max_fixed=3, show_zero_exponent=False, ztol=None): 
    val2 = mpmath.mpc(val)
    real_str = mpmath.nstr(val2.real, sig_digits, strip_zeros=strip_zeros,
                           min_fixed=min_fixed, max_fixed=max_fixed,
                           show_zero_exponent=show_zero_exponent)
    real_str = _check_ztol(val2.real, real_str, ztol)
    imag_str = mpmath.nstr(val2.imag, sig_digits, strip_zeros=strip_zeros,
                           min_fixed=min_fixed, max_fixed=max_fixed,
                           show_zero_exponent=show_zero_exponent)
    imag_str = _check_ztol(val2.imag, imag_str, ztol)
    return real_str, imag_str

def num_str(val, sig_digits=15, strip_zeros=False, min_fixed=-3, max_fixed=3,
            show_zero_exponent=False, ztol=None):
    ret = num_str_pair(val, sig_digits, strip_zeros, min_fixed, max_fixed,
                       show_zero_exponent, ztol)
    imag_str = ret[1]
    if imag_str[0] != '-':
        imag_str = '+' + imag_str
    return ret[0] + imag_str+'j'

def num_str_real(val, sig_digits=15, strip_zeros=False, min_fixed=-4,
                 max_fixed=4, show_zero_exponent=False, ztol=None):
    ret = num_str_pair(val, sig_digits, strip_zeros, min_fixed, max_fixed,
                       show_zero_exponent, ztol)
    return ret[0]

def num_st_imag(val, sig_digits=15, strip_zeros=False, min_fixed=-3,
                max_fixed=3, show_zero_exponent=False, ztol=None):
    ret = num_str_pair(val, sig_digits, strip_zeros, min_fixed, max_fixed,
                       show_zero_exponent, ztol)
    return ret[1]

def num_cmp(a, b, atol, rtol):
    return abs(a-b) <= atol + rtol * abs(b)
