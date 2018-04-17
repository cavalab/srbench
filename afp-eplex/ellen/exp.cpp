/*
* File: exp.cpp
* -------------
* This file implements the EvalState class and the ExpNode class
* hierarchy. The public methods are simple enough that they
* should need no individual documentation.
*/
#include "exp.h"
#include "map.h"
#include "strutils.h"
/* Implementation of the EvalState class */
EvalState::EvalState() {
/* Implemented automatically by Map constructor */
}
EvalState::~EvalState() {
/* Implemented automatically by Map destructor */
}
void EvalState::setValue(string var, int value) {
symbolTable.put(var, value);
}
int EvalState::getValue(string var) {
return symbolTable.get(var);
}
bool EvalState::isDefined(string var) {
return symbolTable.containsKey(var);
}
/*
* Implementation of the base ExpNode class. Neither the
* constructor or destructor requires any code at this level.
*/
ExpNode::ExpNode() {
/* Empty */
}
ExpNode::~ExpNode() {
/* Empty */
}
/*
* Implementation of the ConstantNode subclass. For this
* subclass, the implementation must look up the value in the
* EvalState, passed to eval.
*/
ConstantNode::ConstantNode(int value) {
this->value = value;
}
expTypeT ConstantNode::getType() {
return ConstantType;
}
string ConstantNode::toString() {
return IntegerToString(value);
}
int ConstantNode::eval(EvalState & state) {
return value;
}
int ConstantNode::getValue() {
return value;
}
/*
* Implementation of the IdentifierNode subclass. For this
* subclass, the implementation must look up the value in the
* symbol table passed to eval.
*/
IdentifierNode::IdentifierNode(string name) {
this->name = name;
}
expTypeT IdentifierNode::getType() {
return IdentifierType;
}
string IdentifierNode::toString() {
return name;
}
int IdentifierNode::eval(EvalState & state) {
if (!state.isDefined(name)) Error(name + " is undefined");
return state.getValue(name);
}
string IdentifierNode::getName() {
return name;
}
/*
* Implementation of the CompoundNode subclass. For this subclass,
* the implementation must include explicit code for evaluating each
* of the operators.
*/
CompoundNode::CompoundNode(char oper, expressionT l, expressionT r) {
op = oper;
lhs = l;
rhs = r;
}
CompoundNode::~CompoundNode() {
delete lhs;
delete rhs;
}
expTypeT CompoundNode::getType() {
return CompoundType;
}
string CompoundNode::toString() {
return '(' + lhs->toString() + ' ' + op + ' '
+ rhs->toString() + ')';
}
int CompoundNode::eval(EvalState & state) {
if (op == '=') {
if (lhs->getType() != IdentifierType) {
Error("Illegal variable in assignment");
}
int val = rhs->eval(state);
state.setValue(((IdentifierNode *) lhs)->getName(), val);
return val;
}
int left = lhs->eval(state);
int right = rhs->eval(state);
switch (op) {
case '+': return left + right;
case '-': return left - right;
case '*': return left * right;
case '/': return left / right;
}
Error("Illegal operator in expression");
return 0; /* To avoid the warning message */
}
char CompoundNode::getOp() {
return op;
}
expressionT CompoundNode::getLHS() {
return lhs;
}
expressionT CompoundNode::getRHS() {
return rhs;
}