# pynumwrap
Python package wrapping python and mpmath types behind a common interface.

## Installation

Clone the repository and install with the following commands:

    git clone https://github.com/petersbingham/pynumwrap.git
    cd pynumwrap
    python setup.py install
    
## Dependencies
Third Party Libraries (optional, depending on usage): 
 - numpy
 - mpmath
 - sympy
    
## Usage
This package just wraps similar functionality in sympy, mpmath and numpy, along with the two types, mpmath and standard python behind a common interface. Not all functionality is currently supported, but can be added as required (please submit a PR). To know which functions are supported look in the pynumwrap/\_\_init\_\_.py file.

The type can be changed by calling the `use_python_types` and `use_mpmath_types` functions with the dps passed as an optional parameter. Python types are used as default.

The following example illustrates using the two different types:

```python
>>> import pynumwrap as nw
>>> mat_lst = [[1.,2.],[3.,4.]]
>>> py_mat = nw.matrix(mat_lst)
>>> res = nw.invert(py_mat)
>>> res[0,0]
(-1.9999999999999996+0j)        # Standard python type
>>> nw.use_mpmath_types()
>>> mp_mat = nw.matrix(mat_lst)
>>> res = nw.invert(mp_mat)
>>> res[0,0]
mpf('-2.0')                     # mpmath type
```

