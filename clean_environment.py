# This file reads environment.yml manually, and deletes lines
# containing only: followed by text NOT containing the passed
# algorithm.
import sys

ml = sys.argv[1]

f = open("environment.yml", "r")
f2 = open("clean_environment.yml", "w")

for line in f:
    print_line = True
    if "# only:" in line:
        # ellyn used in several packages, so always
        # assume it:
        if ml not in line and "ellyn" not in line:
            print_line = False

    if print_line:
        print(line, end="", file=f2)
