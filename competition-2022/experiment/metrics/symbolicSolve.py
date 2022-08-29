import sympy as sp

x,y,z,w = sp.symbols('x y z w')

def test_symb_expr(f, g):
    print(f, " == ", g, "?")
    print("=================\n")
    print("simplify(f-g):\n", sp.simplify(f - g))
    print("simplify(f)-simplify(g):\n", sp.simplify(f) - sp.simplify(g))
    print("simplify(f/g):\n", sp.simplify(f/g))
    print("simplify(f)/simplify(g):\n", sp.simplify(f) / sp.simplify(g))

    a, b = sp.symbols('a b')
    h = sp.solve_linear(sp.simplify(f - a*g - b), symbols=[a, b])
    res = 1
    if h[0] == a:
        res = sp.simplify(f - h[1]*g - b)
        print("solve f == a*g + b:\n", sp.simplify(f - h[1]*g - b))
    else:
        res = sp.simplify(f - a*g - h[1])
        print("solve f = a*g + b:\n", sp.simplify(f - a*g - h[1]))
    print(res == 0)
    print("=================\n")

# Spatial Coevolution
# f1 and f2 are exactly the same! f2l is a linear transformation of f2
f1 = 1/(1/(1 + x**(-4)) + 1/(1 + y**(-4)))
f2 = ((x**4)*(y**4) + x**4 + y**4 + 1)/(2*(x**4)*(y**4) + x**4 + y**4)
f2l = 10*f2 + 3 
test_symb_expr(f1,f2)
test_symb_expr(f1,f2l)

# a made up function f3 and a linear transformation f4 
f3 = sp.sin(x) + y*x 
f4 = 10*sp.sin(x) + 10*x*y - 5
test_symb_expr(f3, f4)

# a made up function with four variables and its linear transformation 
f5 = x*y + sp.sin(w) - sp.exp( (x+y)/z )
f6 = 5*x*y + 5*sp.sin(w) - sp.exp(sp.log(5)*x/z)*sp.exp(y/z) + 3
test_symb_expr(f5, f6)

# todo: loop through pairs 
test_symb_expr(f1, f6)
test_symb_expr(f3, f6)
