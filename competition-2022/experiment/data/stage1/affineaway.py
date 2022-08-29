from sympy import simplify, preorder_traversal, Integer, Float
from sympy.utilities.lambdify import lambdify
import numpy as np

grav_law = "6.674*10**(-11) * (m*n)/(r**2)"

a = simplify(grav_law)
b = simplify("-10.3*("+grav_law+")-42")

def get_symbols(f):
  """
  Returns a list containing the names of the symbols (variables) in the input sympy expression

  Parameters
  ----------
  f : sympy Expr
    the sympy expression

  Returns
  -------
  list
    list of symbols in f, sorted alphabetically
  """
  symbols = set()
  for n in preorder_traversal(f):
    if n.is_Symbol:
      symbols.add(str(n))
  return sorted(list(symbols))

def only_affine_transform_away(f, g, num_samples=1000):
  """
  Code to check whether two sympy expressions are one affine transform away.
  This is done by measuring absolute Pearson's correlation.

  Parameters
  ----------
  f : sympy Expr
    the first expression
  g : sympy Expr
    the second expression
  num_samples : int, optional
    the number of samples used to compute the absolute Pearson correlation (default is 1000)

  Returns
  -------
  bool
    whether the absolute Pearson correlation, rounded to the 9th decimal, is 1.
  """

  symbols = get_symbols(f)
  if symbols != get_symbols(g):
    return False

  X = np.random.random(size=(len(symbols),num_samples)) * np.random.randint(100) - np.random.randint(50)
  X = list(X)

  np_f = lambdify(symbols, f, 'numpy')
  np_g = lambdify(symbols, g, 'numpy')

  out_f = np_f(*X)
  out_g = np_g(*X)
  
  corr = np.round(np.abs(np.corrcoef(out_f, out_g)[0,1]),9)
  if corr < 1:
    return False

  return True


same = only_affine_transform_away(a,b)
print(same)