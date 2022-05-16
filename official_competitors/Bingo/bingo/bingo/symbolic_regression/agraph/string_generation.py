"""
Attributes
----------
STACK_PRINT_MAP : dict {int: str}
                  A map of node number to a format string for stack output
LATEX_PRINT_MAP : dict {int: str}
                  A map of node number to a format string for latex output
CONSOLE_PRINT_MAP : dict {int: str}
                  A map of node number to a format string for console output
"""
from bingo.symbolic_regression.agraph.operator_definitions \
    import INTEGER, VARIABLE, CONSTANT, ADDITION, SUBTRACTION, MULTIPLICATION, \
           DIVISION, SIN, COS, SINH, COSH, EXPONENTIAL, LOGARITHM, POWER, ABS, \
           SQRT, SAFE_POWER

STACK_PRINT_MAP = {ADDITION: "({}) + ({})",
                   SUBTRACTION: "({}) - ({})",
                   MULTIPLICATION: "({}) * ({})",
                   DIVISION: "({}) / ({}) ",
                   SIN: "sin ({})",
                   COS: "cos ({})",
                   SINH: "sinh ({})",
                   COSH: "cosh ({})",
                   EXPONENTIAL: "exp ({})",
                   LOGARITHM: "log ({})",
                   POWER: "({}) ^ ({})",
                   ABS: "abs ({})",
                   SQRT: "sqrt ({})",
                   SAFE_POWER: "(|{}|) ^ ({})"}
LATEX_PRINT_MAP = {ADDITION: "{} + {}",
                   SUBTRACTION: "{} - ({})",
                   MULTIPLICATION: "({})({})",
                   DIVISION: "\\frac{{ {} }}{{ {} }}",
                   SIN: "sin{{ {} }}",
                   COS: "cos{{ {} }}",
                   SINH: "sinh{{ {} }}",
                   COSH: "cosh{{ {} }}",
                   EXPONENTIAL: "exp{{ {} }}",
                   LOGARITHM: "log{{ {} }}",
                   POWER: "({})^{{ ({}) }}",
                   ABS: "|{}|",
                   SQRT: "\\sqrt{{ {} }}",
                   SAFE_POWER: "(|{}|)^{{ ({}) }}"}
CONSOLE_PRINT_MAP = {ADDITION: "{} + {}",
                     SUBTRACTION: "{} - ({})",
                     MULTIPLICATION: "({})({})",
                     DIVISION: "({})/({}) ",
                     SIN: "sin({})",
                     COS: "cos({})",
                     SINH: "sinh({})",
                     COSH: "cosh({})",
                     EXPONENTIAL: "exp({})",
                     LOGARITHM: "log({})",
                     POWER: "({})^({})",
                     ABS: "|{}|",
                     SQRT: "sqrt({})",
                     SAFE_POWER: "(|{}|)^({})",}


def get_formatted_string(eq_format, command_array, constants):
    """ Builds a formatted string from command array and constants

    Parameters
    ----------
    eq_format : str
        "stack", "latex", or "console"
    command_array : Nx3 array of int
        stack representation of an equation
    constants : list(float)
        list of numerical constants in the equation

    Returns
    -------
    str
        equation formatted in the way specified
    """
    if eq_format == "stack":
        return _get_stack_string(command_array, constants)

    if eq_format == "latex":
        format_dict = LATEX_PRINT_MAP
    else:  # "console"
        format_dict = CONSOLE_PRINT_MAP
    str_list = []
    for stack_element in command_array:
        tmp_str = _get_formatted_element_string(stack_element, str_list,
                                                format_dict, constants)
        str_list.append(tmp_str)
    return str_list[-1]


def _get_stack_string(command_array, constants):
    tmp_str = ""
    for i, stack_element in enumerate(command_array):
        tmp_str += _get_stack_element_string(i, stack_element, constants)

    return tmp_str


def _get_stack_element_string(command_index, stack_element, constants):
    node, param1, param2 = stack_element
    tmp_str = "(%d) <= " % command_index
    if node == VARIABLE:
        tmp_str += "X_%d" % param1
    elif node == CONSTANT:
        if param1 == -1 or param1 >= len(constants):
            tmp_str += "C"
        else:
            tmp_str += "C_{} = {}".format(param1, constants[param1])
    elif node == INTEGER:
        tmp_str += "{} (integer)".format(param1)
    else:
        tmp_str += STACK_PRINT_MAP[node].format(param1, param2)
    tmp_str += "\n"
    return tmp_str


def _get_formatted_element_string(stack_element, str_list,
                                  format_dict, constants):
    node, param1, param2 = stack_element
    if node == VARIABLE:
        tmp_str = "X_%d" % param1
    elif node == CONSTANT:
        if param1 == -1 or param1 >= len(constants):
            tmp_str = "?"
        else:
            tmp_str = str(constants[param1])
    elif node == INTEGER:
        tmp_str = str(int(param1))
    else:
        tmp_str = format_dict[node].format(str_list[param1], str_list[param2])
    return tmp_str
