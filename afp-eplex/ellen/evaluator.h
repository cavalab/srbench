//evaluator e, made up of parser, expression and symbol table
#ifndef EVALUATOR_H
#define EVALUATOR_H
#include "stdafx.h"
#include "params.h"
#include "data.h"
#include "exprtk.hpp"

struct evaluator {
	typedef exprtk::symbol_table<float> symbol_table_t;
	typedef exprtk::expression<float> expression_t;
	typedef exprtk::parser<float> parser_t;

	symbol_table_t symbol_table;
	expression_t expression;
	parser_t parser;

	void init(params& p,data&d)
	{
		for (unsigned int i=0; i<p.allvars.size();i++)
		{
			symbol_table.add_variable(d.label.at(i),d.dattovar.at(i));
		}

		expression.register_symbol_table(symbol_table);
	}
	
	/*evaluator(){}
	~evaluator(){}*/

};
#endif