# Scikit-like regressor wrapper around the symbolic genetic programming algorithm.
# This module exposes a fit/predict interface and configures the GP training process.

from __future__ import annotations

import time
import random
from copy import deepcopy
from typing import Optional, Union, List, Callable
from enum import Enum


import numpy as np
import importlib
import pandas as pd
import math

try:
    sklearn_base = importlib.import_module("sklearn.base")
    BaseEstimator = getattr(sklearn_base, "BaseEstimator")
    RegressorMixin = getattr(sklearn_base, "RegressorMixin")
    validation = importlib.import_module("sklearn.utils.validation")
    check_X_y = getattr(validation, "check_X_y")
    check_array = getattr(validation, "check_array")
except Exception:
    BaseEstimator = type("BaseEstimator", (), {})
    RegressorMixin = type("RegressorMixin", (), {})

class Variable:
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
        self.value = None

    def setValue(self, value: float):
        self.value = value
        self.initialized = True

    def __str__(self) -> str:
        if not self.initialized:
            return self.name
        else:
            return f"{self.name}({self.value:.2f})"
        
    @staticmethod
    def setVariableValues(variableList: list, values: list) -> None:
        if len(variableList) != len(values):
            raise ValueError("Length of variable list and values list must be the same")

        for i in range(len(variableList)):
            variableList[i].setValue(values[i])

class DataSource:
    def getRow(self, index: int) -> list:
        pass

    def createVariableList(self) -> list:
        pass

class ArrayDataSource(DataSource):
    def __init__(self, X, y, columnNames: list):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.columnNames = list(columnNames)
        self.localSaved = True

        if self.X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if self.y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

    def getRow(self, index: int) -> list:
        if index < 0 or index >= self.X.shape[0]:
            raise IndexError("Index out of range")
        return [index] + self.X[index].tolist() + [self.y[index]]

    def createVariableList(self) -> list:
        return [Variable(name) for name in self.columnNames]

class TaylorFunc:
    @staticmethod
    def x_triangle(t, N, f0=1.0):
        """
        Evaluate the triangle-wave Taylor approximation on the fractional part
        of the input and then add back the integer part.

        Example: for t = 2.64, the fractional part is 0.64 and the result is
        approximately 2.64 again when the fractional mapping returns 0.64.
        """
        x = float(t)
        whole = int(np.floor(x))
        frac = x - whole

        total_sum = 0.0
        for n in range(1, N + 1):
            numerator = np.cos((2 * n - 1) * np.pi * frac)
            denominator = (2 * n - 1) ** 2
            total_sum += numerator / denominator

        constant_factor = 4 / (np.pi ** 2)
        frac_result = 0.5 - (constant_factor * total_sum)
        return float(whole + frac_result)


def get_declining_random_fast(n: int, rate: float = 1.0, rng: random.Random = None) -> int:
    rng = rng if rng is not None else random
    u = rng.random()  # Náhodné číslo od 0.0 do 1.0
    
    # Inverzní transformace pro exponenciální rozdělení
    # (převádí rovnoměrné rozdělení na exponenciální klesající)
    val = -math.log(1 - u * (1 - math.exp(-rate))) / rate
    
    # Přemapujeme hodnotu z rozsahu (0..1) na celé číslo od 1 do n
    result = int(val * n) + 1
    
    # Pojistka pro extrémní případ, kdy random() vrátí přesně 1.0
    return min(result, n)

# Arithmetic function primitives used by the GP framework.
# This module defines simple function wrappers for addition, subtraction, and multiplication.

class Function:
    def __init__(self, formula: Callable, name: str):
        self.formula = formula
        self.name = name

    def __call__(self, x, y):
        return self.formula(x, y)

    # --- BINÁRNÍ FUNKCE ---
    @staticmethod
    def addFunction(): return Function(lambda x, y: x + y, "+")
    
    @staticmethod   
    def subFunction(): return Function(lambda x, y: x - y, "-")    
    
    @staticmethod
    def multFunction(): return Function(lambda x, y: x * y, "*")
    
    @staticmethod
    def divFunction(): 
        # Bezpečné dělení: pokud je jmenovatel blízko nuly, vrátíme 1.0, jinak bezpečně vydělíme
        return Function(lambda x, y: np.where(np.abs(y) < 1e-9, 1.0, x / np.where(np.abs(y) < 1e-9, 1.0, y)), "/")
    
    @staticmethod
    def powFunction(): 
        # Bezpečné umocňování: Základ (x) vynutíme, aby byl nezáporný pomocí np.maximum(x, 0),
        # tím pádem NumPy nikdy nevygeneruje komplexní číslo (imaginární 'j')
        return Function(lambda x, y: np.clip(np.maximum(x, 0.0) ** y, -1e10, 1e10), "^")

    @staticmethod
    def logFunction(): 
        # Bezpečný logaritmus: Vynutíme, aby uvnitř logu byla minimálně hodnota 1e-9,
        # tím pádem logaritmus nikdy neuteče do -nekonečna
        return Function(lambda x, y: np.log(np.maximum(np.abs(x), 1e-9)), "log")

    # --- UNÁRNÍ FUNKCE ---
    @staticmethod
    def sinFunction(): return Function(lambda x, y: np.sin(x), "sin")
    
    @staticmethod
    def cosFunction(): return Function(lambda x, y: np.cos(x), "cos")
    
    @staticmethod
    def tanFunction(): return Function(lambda x, y: np.tan(x), "tan")
    
    @staticmethod
    def expFunction(): 
        return Function(lambda x, y: np.exp(np.clip(x, -100, 100)), "exp")
    
    @staticmethod
    def sqrtFunction(): return Function(lambda x, y: np.sqrt(np.abs(x)), "sqrt")
    
    @staticmethod
    def absFunction(): return Function(lambda x, y: np.abs(x), "abs")
    
    @staticmethod
    def asinFunction(): 
        return Function(lambda x, y: np.arcsin(np.clip(x, -1, 1)), "asin")
    
    @staticmethod
    def acosFunction(): 
        return Function(lambda x, y: np.arccos(np.clip(x, -1, 1)), "acos")
    
    @staticmethod
    def atanFunction(): return Function(lambda x, y: np.arctan(x), "atan")
    
    @staticmethod
    def sinhFunction(): 
        return Function(lambda x, y: np.sinh(np.clip(x, -50, 50)), "sinh")
    
    @staticmethod
    def coshFunction(): 
        return Function(lambda x, y: np.cosh(np.clip(x, -50, 50)), "cosh")
    
    @staticmethod
    def tanhFunction(): return Function(lambda x, y: np.tanh(x), "tanh")

    # --- STATICKÁ MAPA (Inicializuje se pouze jednou při importu třídy) ---
    _registry = {}

    @classmethod
    def decoder(cls, name: str) -> "Function":
        """
        Maximálně efektivní vyhledávání v předpřipravené mapě bez alokace nové paměti.
        """
        clean_name = name.strip().lower()
        if clean_name in cls._registry:
            return cls._registry[clean_name]
        raise ValueError(f"Funkce s názvem '{name}' není v setu SRBench podporována.")

# Naplnění registru jednorázově hned pod třídou
Function._registry = {
    # Binární
    "+": Function.addFunction(), "add": Function.addFunction(),
    "-": Function.subFunction(), "sub": Function.subFunction(), "subb": Function.subFunction(),
    "*": Function.multFunction(), "mul": Function.multFunction(), "mult": Function.multFunction(),
    "/": Function.divFunction(), "div": Function.divFunction(),
    "^": Function.powFunction(), "**": Function.powFunction(), "pow": Function.powFunction(),
    
    # Unární
    "sin": Function.sinFunction(),
    "cos": Function.cosFunction(),
    "tan": Function.tanFunction(),
    "exp": Function.expFunction(),
    "log": Function.logFunction(), "ln": Function.logFunction(),
    "sqrt": Function.sqrtFunction(),
    "abs": Function.absFunction(),
    "asin": Function.asinFunction(), "arcsin": Function.asinFunction(),
    "acos": Function.acosFunction(), "arccos": Function.acosFunction(),
    "atan": Function.atanFunction(), "arctan": Function.atanFunction(),
    "sinh": Function.sinhFunction(),
    "cosh": Function.coshFunction(),
    "tanh": Function.tanhFunction(),
}

class NodeType(Enum):
    FUNCTIONNODE = 1
    TERMINALNODE = 2

class Node:
    def __init__(self, type: NodeType, value: float = None, variable: Variable = None, funcName: str = ""):
        self.type = type
        self.value = value
        self.variable = variable
        self.funcName = funcName

    def isFunctionNode(self):
        return self.type == NodeType.FUNCTIONNODE
    
    def isTerminalNode(self):
        return self.type == NodeType.TERMINALNODE

    def __str__(self) -> str:
        if self.variable is not None:
            return f" {self.variable} "
        if self.value is None:
            return f"{self.type.name}:None"

        if self.isFunctionNode():
            formatted = f"{self.value:.2f}"
            return f"{self.funcName}({formatted})"

        formatted = f"{self.value:.2f}"
        return formatted

    @staticmethod
    def createRandomFunctionNode(minVal: float = 0, maxVal: float = 2, rng: random.Random = None)  -> Node:
        rng = rng if rng is not None else random
        value = rng.uniform(minVal, maxVal)
        return Node(NodeType.FUNCTIONNODE, value)
    
    @staticmethod
    def createRandomTerminalNode(minVal: float = 0, maxVal: float = 2, variableList: list = None, variableProb: float = 0.5, rng: random.Random = None) -> Node:
        rng = rng if rng is not None else random
        value = None
        variable = None
        if variableList is not None and rng.random() < variableProb:
            variable = rng.choice(variableList)
        else:
            value = rng.uniform(minVal, maxVal)
        return Node(NodeType.TERMINALNODE, value, variable)

class SmoothMultifunctionSet:
    """Container for smooth multifunction evaluation and interpolation.

    Stores a mapping of discrete arithmetic functions and optionally applies
    a triangle-wave transformation to the function value before interpolation.
    """

    def __init__(self, name: str = "", minVal: int = 0, maxVal: int = 3, taylorSumElements: int = 100, useTriangleFval: bool = True):
        """Initialize a smooth multifunction set.

        Args:
            name: Human-readable name for the function set.
            minVal: Minimum function index.
            maxVal: Maximum function index.
            taylorSumElements: Number of interpolation samples for triangle mapping.
            useTriangleFval: If True, use triangular f-value mapping instead of raw fval.
        """
        self.name = name
        self.minVal = minVal
        self.maxVal = maxVal
        self.taylorSumElements = taylorSumElements
        self.useTriangleFval = useTriangleFval
        self.functionMap = dict()

    def addFunction(self, func: Callable, val: int) -> None:
        """Add a function object for a specific discrete index value."""
        self.functionMap[val] = func

    def calculateResult(self, x: float, y: float, fval: float, N: int | None = None) -> float:
        """Calculate a smooth result from two operands and an f-value.

        The triangular mapping is applied when useTriangleFval is enabled.
        The final result is a weighted blend of the two nearest functions.
        """
        if N is None:
            N = self.taylorSumElements

        if(fval > self.maxVal):
            print("Val is greater than maxVal")
            raise ValueError()
        
        if(fval < self.minVal):
            print("Val is lower than minVal")
            raise ValueError()
        
        if self.useTriangleFval:
            val = TaylorFunc.x_triangle(fval, N)
        else:
            val = fval

        f1 = self.functionMap.get(int(val))
        # Use modulo maxVal to safely wrap around the circular buffer
        f2 = self.functionMap.get((int(val) + 1) % self.maxVal)

        if f1 is None:
            raise ValueError(f"Function for index {int(val)} not found in functionMap")
        if f2 is None:
            raise ValueError(f"Function for index {(int(val) + 1) % self.maxVal} not found in functionMap")

        fval2 = val - int(val)
        fval1 = 1 - fval2
        
        result1 = fval1 * f1.formula(x, y)
        result2 = fval2 * f2.formula(x, y)

        return result1 + result2
    
    @staticmethod
    def createClassicMultifunctionSet(taylorSumElements: int = 100, useTriangleFval: bool = True) -> SmoothMultifunctionSet:
        """Create a default multifunction set with add, subtract, and multiply."""
        fset = SmoothMultifunctionSet(name = "+,-,*", minVal=0, maxVal=3, taylorSumElements=taylorSumElements, useTriangleFval=useTriangleFval)
        fset.addFunction(Function.addFunction(), 0)
        fset.addFunction(Function.subFunction(), 1)
        fset.addFunction(Function.multFunction(), 2)
        return fset

    @staticmethod
    def createMultifunctionSetByNames(function_names: List[str], taylorSumElements: int = 100, useTriangleFval: bool = True) -> SmoothMultifunctionSet:
        """Dynamically create a SmoothMultifunctionSet based on a list of function names from SRBench."""
        if not function_names:
            raise ValueError("The function list for MultifunctionSet cannot be empty.")

        # Lokální import pro bezpečné rozbití cyklických importů

        set_name = ",".join(function_names)
        num_functions = len(function_names)

        fset = SmoothMultifunctionSet(
            name=set_name, 
            minVal=0, 
            maxVal=num_functions, 
            taylorSumElements=taylorSumElements, 
            useTriangleFval=useTriangleFval
        )

        for index, name in enumerate(function_names):
            try:
                # Voláme tvůj originální, skvělý dekodér!
                decoded_func = Function.decoder(name)
                fset.addFunction(decoded_func, index)
            except ValueError as e:
                print(f"Initialization warning: {e}")
                raise e

        return fset

    @staticmethod
    def createBasicFunctionSet(taylorSumElements: int = 100, useTriangleFval: bool = True) -> SmoothMultifunctionSet:
        """Create a default basic multifunction set for black-box regression tasks."""
        # Lokální import pro jistotu i zde

        blackbox_functions = ["add", "sub", "mul", "div", "pow", "sin", "log"]
        num_functions = len(blackbox_functions)
        
        symbols = ["+", "-", "*", "/", "^", "sin", "log"]
        set_name = ",".join(symbols)

        fset = SmoothMultifunctionSet(
            name=set_name, 
            minVal=0, 
            maxVal=num_functions, 
            taylorSumElements=taylorSumElements, 
            useTriangleFval=useTriangleFval
        )

        for index, name in enumerate(blackbox_functions):
            decoded_func = Function.decoder(name)
            fset.addFunction(decoded_func, index)

        return fset
    

class Individual:
    """A flat vector GP individual for evaluation and genetic operators."""

    def __init__(self, depth: int = 0, variableList: list = None, funcList: list = None):
        """Create a vector-based GP individual with a given depth."""
        self.depth = depth
        self.vector = np.array([], dtype=float)
        self.variableList = variableList
        self.funcList = funcList

    def transformToPrintableIndividual(self) -> PrintableIndividual:
        """Convert the vector individual into a tree-like PrintableIndividual."""
        pInd = PrintableIndividual(depth=self.depth, randomInit=False, variableList=self.variableList)
        pInd.nodes = []
        for i in range(pow(2, self.depth - 1) - 1):
            fname = "F" + str(int(self.vector[(2*i)+1]) + 1)
            pInd.nodes.append(Node(type=NodeType.FUNCTIONNODE, value=self.vector[2*i], funcName=fname))
        for i in range(pow(2, self.depth - 1) - 1, pow(2, self.depth) - 1):
            if int(self.vector[(2 * i) + 1]) == 1: # Variable
                var = self.variableList[int(self.vector[(2*i)])]
                pInd.nodes.append(Node(type=NodeType.TERMINALNODE, variable=var))
            elif int(self.vector[(2 * i) + 1]) == 0: # Constant
                val = self.vector[2*i]
                pInd.nodes.append(Node(type=NodeType.TERMINALNODE, value=val))
            else:
                print(self.vector.at((2 * i) + 1))
                raise ValueError()
        return pInd
            
    def __str__(self):
        """Return a human-readable tree representation of the individual."""
        pInd = self.transformToPrintableIndividual()
        return pInd.getVerticalTreeString()
    
    def printTree(self):
        """Print the tree form of the individual to standard output."""
        pInd = self.transformToPrintableIndividual()
        pInd.printVerticalTree()



    @staticmethod
    def createRandomIndividual(depth: int = 0, variableList: list = None, variableProbability: float = 0.5, functionList: list = None,
                            minTerminalNodeVal: float = 0, maxTerminalNodeVal: float = 10, rng: random.Random = None) -> Individual:
        """Create a new random flat-vector individual for the given depth and variables."""
        rng = rng if rng is not None else random
        newIndividual = Individual(depth=depth, variableList=variableList, funcList=functionList) 
        
        # Dočasný plochý seznam
        temp_list = []
        
        for _ in range(pow(2, depth - 1) - 1):
            fidx = rng.randint(0, len(functionList) - 1)
            fval = rng.uniform(functionList[fidx].minVal, functionList[fidx].maxVal)
            # .extend() přidá prvky z nuly/seznamu jednotlivě za sebou
            temp_list.extend([fval, fidx])  
            
        for _ in range(pow(2, depth - 1)):
            seed = rng.random()
            if seed < variableProbability:
                varIdx = rng.randint(0, len(variableList) - 1)
                temp_list.extend([varIdx, 1.0]) # 1.0 aby seděl datový typ float
            else:
                tval = rng.uniform(minTerminalNodeVal, maxTerminalNodeVal)
                temp_list.extend([tval, 0.0])

        # Výsledkem bude jedno dlouhé ploché 1D NumPy pole
        
        newIndividual.vector = np.array(temp_list, dtype=float)
        return newIndividual
    
    def evaluate(self, fList: list, varList: list) -> float:
        """Evaluate the individual recursively using the given function and variable lists."""
        return self.evaluateRec(fList, varList, 0, 1)
    
    def evaluateRec(self, fList: list, varList: list, idx: int, depth: int) -> float:
        """Recursive helper for evaluate, traversing the flat vector representation."""
        if depth >= self.depth: # Terminal node
            val = self.vector[2*idx]
            type = self.vector[(2*idx)+1]
            if type == 0: # Constant
                return val
            else: # Variable
                return varList[int(val)].value

        else:
            left = self.evaluateRec(fList, varList, 2 * idx + 1, depth + 1)
            right = self.evaluateRec(fList, varList, 2 * idx + 2, depth + 1)

            fIdx = int(self.vector[(2 * idx) + 1])
            fVal = self.vector[2 * idx]
            f = fList[fIdx]

            return f.calculateResult(left, right, fVal)



class PrintableIndividual:
    """A tree-like individual representation used for printing and tree-based mutation."""

    def __init__(self, depth: int = 0, randomInit: bool = False, 
                 variableList: list = None, variableProbability: float = 0.5,  
                 minTerminalNodeVal: int = 0, maxTerminalNodeVal: int = 10,
                 minFunctionNodeVal: int = 0, maxFunctionNodeVal: int = 2, rng: random.Random = None):
        """Initialize a printable individual, optionally randomly filled."""
        self.depth = depth
        self.nodes = []
        self.variableList = variableList  # Uchováme si odkaz na variableList
        rng = rng if rng is not None else random
        
        if randomInit:
            for i in range(pow(2, depth - 1) - 1):
                self.nodes.append(Node.createRandomFunctionNode(minFunctionNodeVal, maxFunctionNodeVal, rng=rng))
            for i in range(pow(2, depth - 1)):
                self.nodes.append(Node.createRandomTerminalNode(minTerminalNodeVal, maxTerminalNodeVal, variableList, variableProbability, rng=rng))
        else:
            for i in range(pow(2, depth) - 1):
                self.nodes.append(None)

    @staticmethod
    def createRandomIndividual(smoothMultifunctionSet: SmoothMultifunctionSet, depth: int = 0, 
                               variableList: list = None, variableProbability: float = 0.5,
                               minTerminalNodeVal = 0, maxTerminalNodeVal = 10) -> Individual:
        """Create a random tree-based individual using the provided function set."""
        newIndividual = Individual(depth, randomInit = True, 
                                   variableList = variableList, variableProbability = variableProbability, 
                                   minTerminalNodeVal = minTerminalNodeVal, maxTerminalNodeVal = maxTerminalNodeVal,
                                   minFunctionNodeVal = smoothMultifunctionSet.minVal, maxFunctionNodeVal = smoothMultifunctionSet.maxVal)
        return newIndividual

    def getTreeString(self, index: int = 0, prefix: str = "", isLast: bool = True) -> str:
        """Return an ASCII tree representation of the individual."""
        if index >= len(self.nodes):
            return ""

        node = self.nodes[index]
        if node is None:
            node_label = "<empty>"
        else:
            node_label = str(node)

        branch = "└── " if isLast else "├── "
        result = prefix + branch + node_label + "\n"

        left_index = 2 * index + 1
        right_index = 2 * index + 2
        if left_index < len(self.nodes) or right_index < len(self.nodes):
            next_prefix = prefix + ("    " if isLast else "│   ")
            if left_index < len(self.nodes):
                result += self.getTreeString(left_index, next_prefix, right_index >= len(self.nodes))
            if right_index < len(self.nodes):
                result += self.getTreeString(right_index, next_prefix, True)

        return result

    def printTree(self) -> None:
        """Print the current tree structure to stdout."""
        if not self.nodes:
            print("<empty individual>")
            return
        if self.nodes[0] is None:
            print("<empty root>")
            return
        print(self.getTreeString())

    def getVerticalTreeString(self) -> str:
        """Construct a vertical ASCII rendering of the tree."""
        if not self.nodes:
            return "<empty individual>\n"

        node_texts = [str(node) if node is not None else "" for node in self.nodes]
        node_widths = [max(3, len(text)) for text in node_texts]
        
        leaf_count = 2 ** max(0, self.depth - 1)
        leaf_start = 2 ** (self.depth - 1) - 1

        # Dynamický výpočet minimální mezery (gap) na základě hloubky stromu.
        # Čím je strom hlubší, tím větší mezery mezi listy dole potřebujeme,
        # aby se nad ně bezpečně vešly texty větších funkčních uzlů.
        base_gap = 1
        if self.depth > 2:
            # Zjistíme maximální šířku jakéhokoliv funkčního uzlu nad listy
            max_fnode_width = max(node_widths[:leaf_start]) if leaf_start > 0 else 3
            # Mezera se odvíjí od velikosti funkčních uzlů a hloubky
            base_gap = max(1, (max_fnode_width // 2) + 1)

        leaf_centers = []
        current_x = 0
        for i in range(leaf_count):
            idx = leaf_start + i
            width = node_widths[idx] if idx < len(node_widths) else 3
            leaf_centers.append(current_x + width // 2)
            
            if i < leaf_count - 1:
                next_idx = leaf_start + i + 1
                next_width = node_widths[next_idx] if next_idx < len(node_widths) else 3
                
                # Mezera se dynamicky přizpůsobuje šířce sousedních listů a vypočtené base_gap
                gap = base_gap + max(0, (width + next_width) // 4)
                current_x += width + gap
            else:
                current_x += width

        total_width = current_x
        centers_by_level = [leaf_centers]

        while len(centers_by_level[0]) > 1:
            current = centers_by_level[0]
            parents = []
            for i in range(0, len(current), 2):
                parents.append((current[i] + current[i + 1]) // 2)
            centers_by_level.insert(0, parents)

        lines = []
        for level in range(self.depth):
            centers = centers_by_level[level]
            node_line = [" "] * total_width

            for j, center in enumerate(centers):
                index = 2 ** level - 1 + j
                text = node_texts[index] if index < len(node_texts) else ""
                width = node_widths[index] if index < len(node_widths) else 3
                text = text.center(width)
                
                # BEZPEČNOSTNÍ POJISTKA: Výpočet levého okraje nesmí klesnout pod nulu
                left = max(0, int(center - width // 2))
                
                for k, ch in enumerate(text):
                    # Zabrání zápisu mimo vyhrazenou šířku řádku
                    if left + k < total_width:
                        node_line[left + k] = ch

            if level > 0:
                indicator_line = [" "] * total_width
                for center in centers:
                    if center < total_width:
                        indicator_line[center] = "|"
                lines.append("".join(indicator_line))

            lines.append("".join(node_line))

            if level < self.depth - 1:
                branch_line = [" "] * total_width
                connector_line = [" "] * total_width
                next_centers = centers_by_level[level + 1]

                for j, center in enumerate(centers):
                    if center < total_width:
                        branch_line[center] = "|"
                    
                    if 2 * j + 1 < len(next_centers):
                        left_child = next_centers[2 * j]
                        right_child = next_centers[2 * j + 1]
                        # Zafixování rozsahu, aby spojovníky nepřetekly šířku plátna
                        start_x = min(total_width - 1, left_child)
                        end_x = min(total_width - 1, right_child)
                        for x in range(start_x, end_x + 1):
                            connector_line[x] = "-"

                lines.append("".join(branch_line))
                lines.append("".join(connector_line))

        return "\n".join(lines) + "\n"

    def printVerticalTree(self) -> None:
        """Print the vertical tree rendering without adding an extra newline."""
        print(self.getVerticalTreeString(), end="")

    def __str__(self) -> str:
        """Return a compact tree-string representation."""
        return self.getTreeString()
    
    def evaluate(self, smoothMultifunctionSet: SmoothMultifunctionSet) -> float:
        """Evaluate the printable individual using the provided smooth multifunction set."""
        if not self.nodes or self.nodes[0] is None:
            raise ValueError("Cannot evaluate an empty individual")

        return self.evaluateRec(smoothMultifunctionSet, 0, 1)
    
    def evaluateRec(self, smoothMultifunctionSet: SmoothMultifunctionSet, idx: int, depth: int) -> float:
        """Recursive evaluation helper for printable individuals."""
        if depth > self.depth - 1:
            node = self.nodes[idx]
            if node is None:
                raise ValueError("Terminal node is None")
            if node.variable:
                if not node.variable.initialized:
                    raise ValueError(f"Variable '{node.variable}' not initialized with a value")
                return node.variable.value
            else:
                return node.value
        else:
            left = self.evaluateRec(smoothMultifunctionSet, 2 * idx + 1, depth + 1)
            right = self.evaluateRec(smoothMultifunctionSet, 2 * idx + 2, depth + 1)
            return smoothMultifunctionSet.calculateResult(left, right, self.nodes[idx].value)
        
    def getConstantNodesVector(self) -> list:
        """Return a list of constant terminal nodes in the individual."""
        constantNodesVector = []
        for node in self.nodes[2 ** (self.depth - 1) - 1:]:
            if node is not None and not node.variable:
                constantNodesVector.append(node)
        return constantNodesVector
    
class MeanSquaredErrorFitnessFunctionVector:
    def evaluateFitness(self, individual: Individual, dataSource: DataSource, dataIndexes: list, 
                        fList: list, variableList: list) -> float:
        if dataSource is None:
            raise Exception("Data source is required for fitness evaluation")

        totalError = 0.0
        numRows = len(dataIndexes)

        for i in range(numRows):
            row = dataSource.getRow(dataIndexes[i])
            inputValues = row[1:-1]
            targetValue = row[-1]
            Variable.setVariableValues(variableList, inputValues)   
            predictedValue = individual.evaluate(fList=fList, varList=variableList)
            error = (predictedValue - targetValue) ** 2
            totalError += error

        meanSquaredError = totalError / numRows

        if meanSquaredError == 0:
            print("Perfect fit found!")
            fitnessValue = float('inf')  # Perfect fit
        else:
            fitnessValue = 1 / meanSquaredError  # Inverse of MSE as fitness

        return fitnessValue

class Mutation():
    """Mutation operators for GP individuals."""

    @staticmethod
    def vectorMutation(mutationRate: float, individual: Individual, fList: list, varlist: list, minTerminalVal: float, maxTerminalVal: float, variableProbability: float, rng: random.Random = None) -> None:
        """Mutate a flat vector individual with given mutation rate and terminals."""
        rng = rng if rng is not None else random
        nodeCnt = individual.vector.size

        for i in range(individual.vector.size):
            seed = rng.random()
            if seed < mutationRate:
                originVal = individual.vector[i]
                if i < ((nodeCnt / 2) - 1): # func node
                    if (i % 2) == 0: # Value
                        f = fList[int(individual.vector[i+1])]
                        newVal = rng.uniform(f.minVal, f.maxVal)
                        individual.vector[i] = newVal
                    else: # Type
                        newType = rng.randint(0, len(fList) - 1)
                        f = fList[newType]
                        newVal = rng.uniform(f.minVal, f.maxVal)
                        individual.vector[i] = newType
                        individual.vector[i - 1] = newVal

                else: # terminal node
                    if (i % 2) == 0:
                        type = int(individual.vector[i + 1])
                        if type == 0: #constant
                            newVal = rng.uniform(minTerminalVal, maxTerminalVal)
                            individual.vector[i] = newVal
                        else: # variable
                            newVal = rng.randint(0, len(varlist) - 1)
                            individual.vector[i] = newVal
                    else:
                        seed = rng.random()
                        if seed >= variableProbability:
                            newVal = rng.uniform(minTerminalVal, maxTerminalVal)
                            individual.vector[i - 1] = newVal
                            individual.vector[i] = 0
                        else:
                            newVal = rng.randint(0, len(varlist) - 1)
                            individual.vector[i - 1] = newVal
                            individual.vector[i] = 1


    @staticmethod
    def nodeMutation(individual: PrintableIndividual, mutationRate: float = 0.1,
               minFunctionNodeVal: float = 0, maxFunctionNodeVal: float = 2,
               minTerminalNodeVal: float = 0, maxTerminalNodeVal: float = 10, 
               variableList: list = None, variableProbability: float = 0.3, rng: random.Random = None) -> None:
        """Mutate a tree-based individual by changing nodes or terminals."""
        rng = rng if rng is not None else random
        originalIndividual = individual.getTreeString()
        for i in range(len(individual.nodes)):
            if rng.random() < mutationRate:
                node = individual.nodes[i]
                if node is not None:
                    if node.isFunctionNode():
                        new_value = rng.uniform(minFunctionNodeVal, maxFunctionNodeVal)
                        individual.nodes[i] = type(node)(node.type, new_value)
                    else:
                        if len(variableList) > 0 and rng.random() < variableProbability:
                            new_variable = rng.choice(variableList)
                            individual.nodes[i] = type(node)(node.type, None, new_variable)
                        else:
                            new_value = rng.uniform(minTerminalNodeVal, maxTerminalNodeVal)
                            individual.nodes[i] = type(node)(node.type, new_value, None)

    @staticmethod
    def vectorOnePointMutation(vectorIndividual: list, mutationRate: float = 0.1,
               minTerminalValue: float = 0, maxTerminalValue: float = 10, rng: random.Random = None) -> None:
        """Apply one-point mutation to a flat vector individual."""
        rng = rng if rng is not None else random
        for i in range(len(vectorIndividual)):
            if rng.random() < mutationRate:
                vectorIndividual[i] = rng.uniform(minTerminalValue, maxTerminalValue)

class Crossover():
    """Crossover operators for recombining GP individuals."""

    @staticmethod
    def swappingPointCrossover(ind1: Individual, ind2: Individual, fset: list, varlist: list, minTerminalVal: float, maxTerminalVal: float, rng: random.Random = None) -> Individual:
        """Create a new individual by swapping vector entries at random points."""
        rng = rng if rng is not None else random
        newInd = deepcopy(ind1)
        nodeCnt = ind1.vector.size
        pointCnt = get_declining_random_fast(n=nodeCnt, rate=18, rng=rng)
        indexes = list(range(0, nodeCnt))
        points = []

        while(pointCnt > 0):    
            seed = rng.randint(0, len(indexes) - 1)
            points.append(indexes[seed])
            indexes.pop(seed)
            pointCnt -= 1

        for point in points:
            newVal = ind2.vector[point]
            if point < ((nodeCnt / 2) - 1): #Function node
                if (point % 2) == 0: #Function value
                    funcIdx = int(newInd.vector[point + 1])
                    f = fset[funcIdx]
                    if newVal >= f.minVal and newVal <= f.maxVal:
                        newInd.vector[point] = newVal
                    else:
                        newInd.vector[point] = rng.uniform(f.minVal, f.maxVal)

                elif (point % 2) == 1: #Function type
                    f = fset[int(newVal)]
                    newInd.vector[point] = newVal
                    fval = newInd.vector[point - 1]
                    if fval < f.minVal or fval > f.maxVal:
                        newInd.vector[point - 1] = rng.uniform(f.minVal, f.maxVal)

            else: # Terminal node
                if (point % 2) == 0: # Terminal value
                    newType = ind2.vector[point + 1]
                    originType = newInd.vector[point + 1]
                    if newType == originType:
                        newInd.vector[point] = newVal
                    else:
                        if originType == 0: # Constant
                            newInd.vector[point] = rng.uniform(minTerminalVal, maxTerminalVal)
                        if originType == 1: # Varible
                            newInd.vector[point] = rng.randint(0, len(varlist) - 1)

                elif (point % 2) == 1: # Terminal type
                    originType = newInd.vector[point]
                    if originType != newVal:
                        newInd.vector[point] = newVal
                        if newVal == 0: #constant
                            newInd.vector[point - 1] = rng.uniform(minTerminalVal, maxTerminalVal)
                        elif newVal == 1: #variable 
                            newInd.vector[point - 1] = rng.randint(0, len(varlist) - 1)

        return newInd
    
    @staticmethod
    def betweenPointCrossover(ind1: Individual, ind2: Individual, fset: list, varlist: list, minTerminalVal: float, maxTerminalVal: float, rng: random.Random = None, percent: float = 10.0) -> Individual:
        """Create a new individual by swapping vector entries at random points."""
        rng = rng if rng is not None else random
        newInd = deepcopy(ind1)
        nodeCnt = ind1.vector.size
        pointCnt = get_declining_random_fast(n=nodeCnt, rate=18, rng=rng)
        indexes = list(range(0, nodeCnt))
        points = []

        def sample_in_interval(origin_val: float, candidate_val: float, lower_bound: float, upper_bound: float) -> float:
            lower = min(origin_val, candidate_val) * (1 - (percent / 100.0))
            upper = max(origin_val, candidate_val) * (1 + (percent / 100.0))
            lower = max(lower_bound, lower)
            upper = min(upper_bound, upper)
            if lower >= upper:
                return max(lower_bound, min(upper_bound, (origin_val + candidate_val) / 2.0))
            return rng.uniform(lower, upper)

        while(pointCnt > 0):    
            seed = rng.randint(0, len(indexes) - 1)
            points.append(indexes[seed])
            indexes.pop(seed)
            pointCnt -= 1

        for point in points:
            newVal = ind2.vector[point]
            if point < ((nodeCnt / 2) - 1): #Function node
                if (point % 2) == 0: #Function value
                    funcIdx = int(newInd.vector[point + 1])
                    originFuncIdx = int(ind2.vector[point + 1])
                    f = fset[funcIdx]
                    originVal = float(newInd.vector[point])
                    candidateVal = float(newVal)
                    if funcIdx == originFuncIdx:
                        newInd.vector[point] = sample_in_interval(originVal, candidateVal, f.minVal, f.maxVal)
                    else:
                        newInd.vector[point] = sample_in_interval(originVal, candidateVal, f.minVal, f.maxVal)

                elif (point % 2) == 1: #Function type
                    f = fset[int(newVal)]
                    newInd.vector[point] = newVal
                    fval = newInd.vector[point - 1]
                    if fval < f.minVal or fval > f.maxVal:
                        newInd.vector[point - 1] = rng.uniform(f.minVal, f.maxVal)

            else: # Terminal node
                if (point % 2) == 0: # Terminal value
                    newType = ind2.vector[point + 1]
                    originType = newInd.vector[point + 1]
                    if newType == originType:
                        if originType == 0: # Constant:
                            originVal = float(newInd.vector[point])
                            candidateVal = float(newVal)
                            newInd.vector[point] = sample_in_interval(originVal, candidateVal, minTerminalVal, maxTerminalVal)
                        else: # Variable
                            newInd.vector[point] = newVal
                    else:
                        if originType == 0: # Constant
                            newInd.vector[point] = rng.uniform(minTerminalVal, maxTerminalVal)
                        if originType == 1: # Varible
                            newInd.vector[point] = rng.randint(0, len(varlist) - 1)

                elif (point % 2) == 1: # Terminal type
                    originType = newInd.vector[point]
                    if originType != newVal:
                        newInd.vector[point] = newVal
                        if newVal == 0: #constant
                            newInd.vector[point - 1] = rng.uniform(minTerminalVal, maxTerminalVal)
                        elif newVal == 1: #variable 
                            newInd.vector[point - 1] = rng.randint(0, len(varlist) - 1)

        return newInd
    
    @staticmethod
    def onePointCrossover(ind1: PrintableIndividual, ind2: PrintableIndividual, rng: random.Random = None) -> tuple[Individual, Individual]:
        """Perform one-point subtree crossover on two printable individuals."""
        rng = rng if rng is not None else random
        if not ind1.nodes or not ind2.nodes:
            return ind1, ind2

        min_nodes = min(len(ind1.nodes), len(ind2.nodes))
        first_leaf_index = (min_nodes - 1) // 2
        if first_leaf_index <= 1:
            crossover_root = 0
        else:
            crossover_root = rng.randint(1, first_leaf_index - 1)

        def subtree_indices(nodes_length: int, root_index: int) -> list[int]:
            indices = [root_index]
            i = 0
            while i < len(indices):
                current = indices[i]
                left = 2 * current + 1
                right = 2 * current + 2
                if left < nodes_length:
                    indices.append(left)
                if right < nodes_length:
                    indices.append(right)
                i += 1
            return indices

        swap_indices = subtree_indices(min_nodes, crossover_root)

        new_ind1 = deepcopy(ind1)
        new_ind2 = deepcopy(ind2)

        # Obnovit originální variableList aby se zachovaly reference na Variable objekty
        # (deepcopy je zkopíroval, ale chceme sdílené originály)
        new_ind1.variableList = ind1.variableList
        new_ind2.variableList = ind2.variableList
        
        # Aktualizovat všechny nody tak aby ukazovaly na originální Variable objekty
        for node in new_ind1.nodes:
            if node is not None and node.variable is not None and hasattr(node.variable, 'name'):
                # Najít odpovídající Variable v originálním variableList
                for orig_var in ind1.variableList:
                    if orig_var.name == node.variable.name:
                        node.variable = orig_var
                        break
        
        for node in new_ind2.nodes:
            if node is not None and node.variable is not None and hasattr(node.variable, 'name'):
                # Najít odpovídající Variable v originálním variableList
                for orig_var in ind2.variableList:
                    if orig_var.name == node.variable.name:
                        node.variable = orig_var
                        break

        for idx in swap_indices:
            new_ind1.nodes[idx], new_ind2.nodes[idx] = new_ind2.nodes[idx], new_ind1.nodes[idx]

        return new_ind1, new_ind2
    
    @staticmethod
    def vectorOnePointCrossover(parent1: list, parent2: list, rng: random.Random = None) -> tuple[list, list]:
        """Perform a one-point crossover on two flat vector parents."""
        rng = rng if rng is not None else random
        if len(parent1) != len(parent2):
            raise ValueError("Parents must be of the same length for one-point crossover.")
        
        crossover_point = rng.randint(0, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return child1, child2
                    

class MeanSquaredErrorFitnessFunction:
    def evaluateFitness(self, individual: Individual, dataSource: DataSource, dataIndexes: list, 
                        smoothmultifunctionset: SmoothMultifunctionSet, variableList: list) -> float:
        if dataSource is None:
            raise Exception("Data source is required for fitness evaluation")

        totalError = 0.0
        numRows = len(dataIndexes)

        for i in range(numRows):
            row = dataSource.getRow(dataIndexes[i])
            inputValues = row[1:-1]
            targetValue = row[-1]
            Variable.setVariableValues(variableList, inputValues)   
            predictedValue = individual.evaluate(smoothmultifunctionset)
            error = (predictedValue - targetValue) ** 2
            totalError += error

        meanSquaredError = totalError / numRows

        if meanSquaredError == 0:
            print("Perfect fit found!")
            fitnessValue = float('inf')  # Perfect fit
        else:
            fitnessValue = 1 / meanSquaredError  # Inverse of MSE as fitness

        return fitnessValue

class Population:
    def __init__(self, smoothMultifunctionSet: SmoothMultifunctionSet = None):
        self.individualList = []
        self.smoothMultifunctionSet = smoothMultifunctionSet

    def initializePopulationFullRandomMethod(self, populationSize: int = 0, depth: int = 0, funcList: list = [],
                                             variableList: list = None, variableProbability: float = 0.5,
                                             minTerminalNodeVal = 0, maxTerminalNodeVal = 10, rng: random.Random = None) -> None:
        for _ in range(populationSize):
            newRandomIndividual = Individual.createRandomIndividual(depth = depth, variableList= variableList, variableProbability=variableProbability,
                                                                    minTerminalNodeVal=minTerminalNodeVal, maxTerminalNodeVal=maxTerminalNodeVal, functionList=funcList, rng=rng)
            self.individualList.append(newRandomIndividual)

class VectorEvolutionAlgorithm:
    """A vector-based genetic programming evolution engine.

    Uses flat vector individuals, crossover, and mutation operators to evolve
    a population of candidate solutions.
    """

    def __init__(self, fList: list, dataSource: DataSource = None, fitnessFunction = None, 
                 dataIndexes: list = None, mutationFunc: Callable = None, crossoverFunc: Callable = None,
                 rng: random.Random = None, taylorSumElements: int = 100, useTriangleFval: bool = True):
        """Initialize vector evolution settings and propagate function set parameters."""
        self.population = Population()
        self.fList = fList
        self.fitnessValues = []
        self.dataSource = dataSource
        self.variableList = None
        self.fitnessFunction = fitnessFunction
        self.dataIndexes = dataIndexes
        self.mutationFunc = mutationFunc
        self.crossoverFunc = crossoverFunc
        self.rng = rng if rng is not None else random.Random()
        self.taylorSumElements = taylorSumElements
        self.useTriangleFval = useTriangleFval
        for func in self.fList:
            if hasattr(func, "taylorSumElements"):
                func.taylorSumElements = taylorSumElements
            if hasattr(func, "useTriangleFval"):
                func.useTriangleFval = useTriangleFval

        for func in self.fList:
            if hasattr(func, "useTriangleFval") and func.useTriangleFval != useTriangleFval:
                raise ValueError(
                    f"Unable to configure useTriangleFval={useTriangleFval} on function set {func}"
                )

    def initFitnessValues(self, populationSize: int = 0):
        """Initialize the fitness value list for the population."""
        self.fitnessValues = [-1] * populationSize

    def runEvolution(self, maxGenerations: int = 100, populationSize: int = 100, depth: int = 4, mutationRate: float = 0.01,
                randomIndividualRate: float = 0.1, variableProbability: float = 0.5,
                minTerminalNodeVal: float = 0, maxTerminalNodeVal: float = 10, maxSeconds: float | None = None) -> Individual | None:
        """Run the vector evolution loop and return the best found individual."""
        self.variableList = self.dataSource.createVariableList()
        self.population.initializePopulationFullRandomMethod(populationSize = populationSize, depth = depth, funcList= self.fList,
                                                             variableList = self.variableList, variableProbability = variableProbability,
                                                             minTerminalNodeVal = minTerminalNodeVal, maxTerminalNodeVal = maxTerminalNodeVal,
                                                             rng=self.rng)
        self.initFitnessValues(populationSize = populationSize)

        best_fitness = float('-inf')
        best_individual = None
        best_generation = None
        start_time = time.monotonic() if maxSeconds is not None else None

        for generation in range(maxGenerations):
            print("------------------------ Generation: " + str(generation) + " ------------------------")
            
            #Fitness evaluation
            for i in range(populationSize):
                self.fitnessValues[i] = self.fitnessFunction.evaluateFitness(individual = self.population.individualList[i], 
                        dataSource = self.dataSource, dataIndexes = self.dataIndexes, fList = self.fList, variableList = self.variableList)

            # Summary statistics for fitness
            if populationSize > 0:
                max_fitness = max(self.fitnessValues)
                min_fitness = min(self.fitnessValues)
                avg_fitness = sum(self.fitnessValues) / float(populationSize)
                print(f"Fitness summary — max: {max_fitness:.4f}, min: {min_fitness:.4f}, avg: {avg_fitness:.4f}")

                # Update best individual seen so far
                try:
                    best_idx = self.fitnessValues.index(max_fitness)
                    candidate = deepcopy(self.population.individualList[best_idx])
                    if max_fitness > best_fitness:
                        best_fitness = max_fitness
                        best_individual = candidate
                        best_generation = generation
                    if math.isinf(max_fitness):
                        print(f"Perfect fit found in GP at generation {generation}. Stopping evolution.")
                        return candidate
                except Exception:
                    pass

            #Selection
            newIndividualList = []
            sortedIndexes = sorted(range(populationSize), key=lambda k: self.fitnessValues[k], reverse=True)

            for i in range(0, populationSize):
                idx1 = sortedIndexes[i] 
                idx2 = sortedIndexes[(i+1) % populationSize]  
                original = self.population.individualList[idx1]
                partner = self.population.individualList[idx2]
                new = self.crossoverFunc(ind1=original, ind2=partner, fset=self.fList, varlist=self.variableList, 
                                         minTerminalVal=minTerminalNodeVal, maxTerminalVal=maxTerminalNodeVal, rng=self.rng)
                scoreOriginal = self.fitnessValues[idx1]
                scoreNew = self.fitnessFunction.evaluateFitness(individual = new, 
                        dataSource = self.dataSource, dataIndexes = self.dataIndexes, fList = self.fList, variableList = self.variableList)
                if scoreNew >= scoreOriginal:
                    newIndividualList.append(new)
                else:
                    newIndividualList.append(original)

            randomIndividualCount = int(round(randomIndividualRate * populationSize))
            for idx in range(randomIndividualCount):
                pos = max(0, populationSize - 1 - idx)
                new = Individual.createRandomIndividual(
                    depth=depth,
                    variableList=self.variableList,
                    variableProbability=variableProbability,
                    functionList=self.fList,
                    minTerminalNodeVal=minTerminalNodeVal,
                    maxTerminalNodeVal=maxTerminalNodeVal,
                    rng=self.rng,
                )
                newIndividualList[pos] = new

            #Mutation
            for i in range(populationSize):
                mutated = deepcopy(self.population.individualList[i])
                self.mutationFunc(individual = mutated, mutationRate = mutationRate, fList = self.fList,
                             minTerminalVal = minTerminalNodeVal, maxTerminalVal = maxTerminalNodeVal,
                             varlist = self.variableList, variableProbability = variableProbability, rng=self.rng)
                originalFitness = self.fitnessValues[i]
                mutatedFitness = self.fitnessFunction.evaluateFitness(individual = mutated, 
                        dataSource = self.dataSource, dataIndexes = self.dataIndexes, fList = self.fList, variableList = self.variableList)
                if mutatedFitness > originalFitness:
                    newIndividualList[i] = mutated

            self.population.individualList = newIndividualList

            if maxSeconds is not None and start_time is not None and (time.monotonic() - start_time) >= maxSeconds:
                print("Stopping evolution because maxSeconds was exceeded.")
                break

        # Po dokončení všech generací vrať nejlepšího jedince
        if best_individual is not None:
            print("\n=== Best individual found ===")
            print(f"Generation: {best_generation}, Fitness: {best_fitness:.6f}")
            try:
                best_individual.printVerticalTree()
            except Exception:
                print(str(best_individual))
        return best_individual


def check_X_y(X, y, dtype=None):
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array or matrix")
    if y_arr.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    return X_arr, y_arr

def check_array(X, dtype=None):
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array or matrix")
    return X_arr

class SMGPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        random_state: Optional[int] = None,
        max_time: Optional[float] = None,
        population_size: int = 100,
        generations: int = 100,
        depth: int = 4,
        mutation_rate: float = 0.01,
        random_individual_rate: float = 0.1,
        variable_probability: float = 0.4,
        min_terminal_node_val: float = 0.0,
        max_terminal_node_val: float = 10.0,
        function_set: Optional[Union[SmoothMultifunctionSet, List[str]]] = None,
        taylor_sum_elements: int = 100,
        use_triangle_fval: bool = True,
        verbose: bool = False,
    ):
        self.random_state = random_state
        self.max_time = max_time
        self.population_size = population_size
        self.generations = generations
        self.depth = depth
        self.mutation_rate = mutation_rate
        self.random_individual_rate = random_individual_rate
        self.variable_probability = variable_probability
        self.min_terminal_node_val = min_terminal_node_val
        self.max_terminal_node_val = max_terminal_node_val
        self.taylor_sum_elements = taylor_sum_elements
        self.use_triangle_fval = use_triangle_fval
        self.function_set = function_set
        self.verbose = verbose

        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.best_individual_ = None
        self.best_fitness_ = None
        self.model_ = None
        self.history_ = []
        self.rng = random.Random(self.random_state) if self.random_state is not None else random.Random()

    def _check_dataframe_columns(self, X):
        if hasattr(X, "columns"):
            return list(X.columns)
        return None

    def _prepare_random_state(self):
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            self.rng = random.Random(self.random_state)
        else:
            self.rng = random.Random()
        
        # Configure or create the SmoothMultifunctionSet dynamically during fit
        if self.function_set is None:
            # Default to the Basic blackbox function set instead of Classic
            self.function_set = SmoothMultifunctionSet.createBasicFunctionSet(
                taylorSumElements=self.taylor_sum_elements, 
                useTriangleFval=self.use_triangle_fval
            )
        elif isinstance(self.function_set, list):
            # If function_set is a list of strings from SRBench, build it dynamically
            self.function_set = SmoothMultifunctionSet.createMultifunctionSetByNames(
                function_names=self.function_set,
                taylorSumElements=self.taylor_sum_elements,
                useTriangleFval=self.use_triangle_fval
            )
        elif isinstance(self.function_set, SmoothMultifunctionSet):
            # If already an object, update its Taylor parameters to match current wrapper configuration
            self.function_set.taylorSumElements = self.taylor_sum_elements
            self.function_set.useTriangleFval = self.use_triangle_fval

    def _create_variable_list(self, feature_names: List[str]) -> List[Variable]:
        return [Variable(name) for name in feature_names]

    def _fitness(self, individual: Individual, X: np.ndarray, y: np.ndarray, function_list: list, variable_list: List[Variable]) -> float:
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("Training data must contain at least one sample.")

        total_error = 0.0
        for row_idx in range(n_samples):
            Variable.setVariableValues(variable_list, X[row_idx].tolist())
            predicted = individual.evaluate(function_list, variable_list)
            total_error += float((predicted - y[row_idx]) ** 2)

        mse = total_error / float(n_samples)
        if mse == 0.0:
            return float('inf')
        return 1.0 / mse

    def _evaluate(self, individual: Individual, X: np.ndarray, variable_list: List[Variable], function_list: list) -> np.ndarray:
        predictions = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            Variable.setVariableValues(variable_list, X[i].tolist())
            predictions[i] = individual.evaluate(function_list, variable_list)
        return predictions

    def _function_list(self) -> list:
        return [self.function_set]

    def _to_sympy_string(self, individual: Individual, feature_names: List[str], fset: SmoothMultifunctionSet) -> str:
        max_val = fset.maxVal

        def op_str(idx_func: int, left: str, right: str) -> str:
            func_obj = fset.functionMap.get(idx_func)
            if func_obj is None:
                raise ValueError(f"Unsupported function index {idx_func}")
            
            name = func_obj.name

            if name == "+": return f"({left} + {right})"
            if name == "-": return f"({left} - {right})"
            if name == "*": return f"({left}*{right})"
            if name == "/": return f"({left}/{right})"
            if name == "^": return f"({left}**{right})"
            
            if name in ["sin", "cos", "tan", "exp", "log", "sqrt", "abs", 
                        "asin", "acos", "atan", "sinh", "cosh", "tanh"]:
                return f"{name}({left})"
                
            raise ValueError(f"Unknown operator name '{name}' at index {idx_func}")

        def node_expr(idx: int, depth: int) -> str:
            if depth >= individual.depth:
                value = individual.vector[2 * idx]
                type_flag = int(individual.vector[2 * idx + 1])
                if type_flag == 0:
                    return f"{float(value):.12g}"
                return feature_names[int(value)]

            left_expr = node_expr(2 * idx + 1, depth + 1)
            right_expr = node_expr(2 * idx + 2, depth + 1)
            
            fval = float(individual.vector[2 * idx])
            
            if fset.useTriangleFval:
                val = TaylorFunc.x_triangle(fval, fset.taylorSumElements)
            else:
                val = fval

            func_idx = int(val)
            next_idx = (func_idx + 1) % max_val
            frac = val - func_idx

            first_expr = op_str(func_idx, left_expr, right_expr)
            if frac == 0.0:
                return first_expr
                
            second_expr = op_str(next_idx, left_expr, right_expr)
            return f"(({1 - frac:.12g})*{first_expr} + {frac:.12g}*{second_expr})"

        return node_expr(0, 1)

    def fit(self, X: Union[np.ndarray, List[List[float]], object], y: Union[np.ndarray, List[float]]) -> 'SMGPRegressor':
        X_arr, y_arr = check_X_y(X, y, dtype=[np.float64, np.float32])
        feature_names = self._check_dataframe_columns(X) or [f"x_{i}" for i in range(X_arr.shape[1])]
        self.feature_names_in_ = feature_names
        self.n_features_in_ = X_arr.shape[1]

        if self.n_features_in_ == 0:
            raise ValueError("At least one feature is required.")
        if self.generations < 1:
            raise ValueError("generations must be at least 1")
        if self.population_size < 1:
            raise ValueError("population_size must be at least 1")

        self._prepare_random_state()
        variable_list = self._create_variable_list(feature_names)
        function_list = [self.function_set]

        data_source = ArrayDataSource(X_arr, y_arr, feature_names)
        algorithm = VectorEvolutionAlgorithm(
            fList=function_list,
            dataSource=data_source,
            fitnessFunction=MeanSquaredErrorFitnessFunctionVector(),
            dataIndexes=list(range(X_arr.shape[0])),
            mutationFunc=Mutation.vectorMutation,
             crossoverFunc=Crossover.swappingPointCrossover,
            rng=self.rng,
            taylorSumElements=self.taylor_sum_elements,
            useTriangleFval=self.use_triangle_fval,
        )

        best_individual = algorithm.runEvolution(
            maxGenerations=self.generations,
            populationSize=self.population_size,
            depth=self.depth,
            mutationRate=self.mutation_rate,
            randomIndividualRate=self.random_individual_rate,
            variableProbability=self.variable_probability,
            minTerminalNodeVal=self.min_terminal_node_val,
            maxTerminalNodeVal=self.max_terminal_node_val,
            maxSeconds=self.max_time,
        )

        if best_individual is None:
            raise RuntimeError("No individual was found during fit.")

        self.best_individual_ = best_individual
        self.best_fitness_ = self._fitness(best_individual, X_arr, y_arr, function_list, variable_list)
        
        # Correctly passing the configured function_set object into the sympy transformation method
        self.model_ = self._to_sympy_string(best_individual, self.feature_names_in_, self.function_set)
        return self

    def predict(self, X: Union[np.ndarray, List[List[float]], object]) -> np.ndarray:
        if self.best_individual_ is None:
            raise ValueError("Estimator is not fitted yet. Call fit() before predict().")

        X_arr = check_array(X, dtype=[np.float64, np.float32])
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, but this estimator was fitted with {self.n_features_in_} features.")

        variable_list = self._create_variable_list(self.feature_names_in_)
        function_list = self._function_list()
        return self._evaluate(self.best_individual_, X_arr, variable_list, function_list)


def model(est: SMGPRegressor, X=None):
    model_str = getattr(est, 'model_', None)
    if model_str is None:
        raise ValueError('Estimator has no model_ string. Fit the estimator first.')
    if X is None:
        return model_str

    if hasattr(X, 'columns'):
        mapping = {f'x_{i}': name for i, name in enumerate(X.columns)}
        result = model_str
        for old_name, new_name in reversed(list(mapping.items())):
            result = result.replace(old_name, new_name)
        return result

    return model_str   
    

"""
def run_srbench_integration_test():
    print("=" * 60)
    print("STARTING SRBENCH INTEGRATION TEST FOR SMGPREGRESSOR")
    print("=" * 60)

    # 1. GENEROVÁNÍ DAT (Přesně tak, jak je posílá SRBench - Pandas DataFrame)
    print("\n[1/5] Generating synthetic benchmark data...")
    np.random.seed(42)
    n_samples = 150
    
    # Vytvoříme 3 proměnné s reálnými názvy
    X_df = pd.DataFrame({
        "temperature": np.random.uniform(-2.0, 2.0, n_samples),
        "pressure": np.random.uniform(0.5, 5.0, n_samples),
        "humidity": np.random.uniform(10.0, 90.0, n_samples)
    })
    
    # Cílová funkce (target), kterou by měl algoritmus aproximovat
    # Použijeme kombinaci operací z našeho registru
    y_true = (X_df["temperature"] * X_df["pressure"]) + np.sin(X_df["temperature"]) - (X_df["humidity"] / 20.0)
    # Přidáme drobný šum, jak je v bencharcích zvykem
    y_arr = y_true + np.random.normal(0, 0.05, n_samples)

    print(f"-> Generated {n_samples} samples with features: {list(X_df.columns)}")

    # 2. INICIALIZACE REGRESORU
    print("\n[2/5] Initializing SMGPRegressor...")
    # Použijeme náš doporučený default set (Basic), hloubku 5 a rychlého Taylora (5 elementů)
    regressor = SMGPRegressor(
        random_state=42,
        max_time=30.0,            # Časový limit 30 sekund
        population_size=150,
        generations=100,          # Pro rychlý test stačí 100 generací
        depth=5,                  # Doporučená hloubka pro benchmarky
        mutation_rate=0.05,
        random_individual_rate=0.1,
        variable_probability=0.4,
        taylor_sum_elements=5,
        use_triangle_fval=True,
        verbose=True              # Chceme vidět průběh výpočtu
    )

    # 3. TRÉNOVÁNÍ (FIT)
    print("\n[3/5] Training the model via .fit()...")
    try:
        regressor.fit(X_df, y_arr)
        print("-> Training finished successfully!")
        print(f"-> Best Fitness achieved: {regressor.best_fitness_:.6g}")
    except Exception as e:
        print(f"!!! CRITICAL ERROR DURING FIT: {e}")
        return

    # 4. PŘEDPOVĚĎ (PREDICT)
    print("\n[4/5] Testing predictions via .predict()...")
    try:
        predictions = regressor.predict(X_df)
        print(f"-> Predictions shape: {predictions.shape}")
        mse = np.mean((predictions - y_arr) ** 2)
        print(f"-> Calculated Mean Squared Error on training data: {mse:.6g}")
        
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            print("!!! WARNING: Predictions contain NaN or Inf values!")
        else:
            print("-> Prediction sanity check: PASSED (No NaNs/Infs)")
    except Exception as e:
        print(f"!!! CRITICAL ERROR DURING PREDICT: {e}")
        return

    # 5. EXPORT MODELU (SYM PY STRING)
    print("\n[5/5] Exporting model string for SymPy compatibility...")
    try:
        # Volání přesně tak, jak to dělá SRBench (předává model a DataFrame)
        raw_model_str = model(regressor)
        final_model_str = model(regressor, X=X_df)
        
        print(f"-> Raw model string (internal names): {raw_model_str}")
        print(f"-> Final SymPy model string (mapped names): {final_model_str}")
        
        # Kontrola, zda se správně nahradily proměnné
        if "x_0" in final_model_str or "x_1" in final_model_str:
            print("!!! WARNING: Variable mapping failed! Internal 'x_i' names are still present.")
        else:
            print("-> Variable mapping check: PASSED (Real column names successfully applied)")

        # Pokus o parsování v SymPy (pokud ho máš nainstalovaný)
        try:
            import sympy as sp
            parsed_expr = sp.parse_expr(final_model_str)
            print("-> SymPy Parsing: PASSED")
            print(f"-> SymPy Simplified: {sp.simplify(parsed_expr)}")
        except ImportError:
            print("-> SymPy library not installed locally, skipping text mathematical parsing validation.")
        except Exception as sympy_err:
            print(f"!!! SymPy Parsing FAILED: {sympy_err}")
            print("Make sure your operators are strictly written in python standard (e.g. log, sin, **).")

    except Exception as e:
        print(f"!!! CRITICAL ERROR DURING MODEL EXPORT: {e}")
        return

    print("\n" + "=" * 60)
    print("INTEGRATION TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    
run_srbench_integration_test()

"""
