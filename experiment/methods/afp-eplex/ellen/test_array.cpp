#include <iostream>
#include <vector>
using namespace std;

void line_to_py(vector<node>& line,bp::list& prog){
	// converts program to tuple for export to python.

	for (auto n: line){
		switch(n.type)
		{
		case 'n':
			prog.append(bp::tuple(n.varname, n.arity, n.value));
			break;
		case 'v':
			prog.append(bp::tuple("k", n.arity, n.value));
			break;
		default:
			prog.append(bp::tuple(n.type, n.arity));
			break;
		}
	}

}
