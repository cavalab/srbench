/*
* File: exp.h
* -----------
* This interface defines a class hierarchy for expressions,
* which allows the client to represent and manipulate simple
* binary expression trees. The primary type exported by this
* file is the pointer type expressionT. The details of the
* individual nodes are relevant only to the implementation.
* This interface also exports a class called EvalState, which
* keeps track of additional information required by the
* evaluator, such as the values of variables.
*/
#pragma once
#ifndef _exp_h
#define _exp_h
#include "stdafx.h"

/*
* Type: expressionT
* -----------------
* For clients, the most important type exported by this interface
* is the expressionT type, which is defined as a pointer to an
* ExpNode object. This is the type used by all other functions
* and methods in the expression package.
*/
class ExpNode;
typedef ExpNode *expressionT;
/*
* Class: EvalState
* ----------------
* This class is passed by reference through the recursive levels
* of the evaluator and contains information from the evaluation
* environment that the evaluator may need to know. The only
* such information implemented here is a symbol table that maps
* variable names into their values.
*/
class EvalState {
public:
/*
* Constructor: EvalState
* Usage: EvalState state;
* -----------------------
* Creates a new EvalState object with no variable bindings.
*/
EvalState();
/*
* Destructor: ~EvalState
* Usage: usually implicit
* -----------------------
* Frees all heap storage associated with this object.
*/
~EvalState();
/*
* Method: setValue
* Usage: state.setValue(var, value);
* ----------------------------------
* Sets the value associated with the specified var.
*/
void setValue(std::string var, float value);
/*
* Method: getValue
* Usage: float value = state.getValue(var);
* ---------------------------------------
* Returns the value associated with the specified variable.
*/
float getValue(std::string var);
/*
* Method: isDefined
* Usage: if (state.isDefined(var)) . . .
* --------------------------------------
* Returns true if the specified variable is defined.
*/
bool isDefined(std::string var);
private:
std::Map<float> symbolTable;
};
/*
* Type: expTypeT
* --------------
* This enumerated type is used to differentiate the three
* different expression types: ConstantType, IdentifierType,
* and CompoundType.
*/
enum expTypeT { ConstantType, IdentifierType, CompoundType };
/*
* Class: ExpNode
* --------------
* This class is used to represent a node in an expression tree.
* ExpNode is an example of an abstract class, which defines the
* structure and behavior of a set of classes but has no objects
* of its own. Any object must be one of the three concrete
* subclasses of ExpNode:
*
* 1. ConstantNode -- an integer constant
* 2. IdentifierNode -- a string representing an identifier
* 3. CompoundNode -- two expressions combined by an operator
*
* The ExpNode class defines the interface common to all ExpNode
* objects; each subclass provides its own specific implementation
* of the common interface.
*
* Note on syntax: Each of the virtual methods in the ExpNode class
* is marked with the designation = 0 on the prototype line. This
* notation is used in C++ to indicate that this method is purely
* virtual and will always be supplied by the subclass.
*/
class ExpNode {
public:
/*
* Constructor: ExpNode
* --------------------
* The base class constructor is empty. Each subclass must provide
* its own constructor.
*/
ExpNode();
/*
* Destructor: ~ExpNode
* Usage: delete exp;
* ------------------
* The destructor deallocates the storage for this expression.
* It must be declared virtual to ensure that the correct subclass
* destructor is called when deleting an expression.
*/
virtual ~ExpNode();
/*
* Method: getType
* Usage: type = exp->getType();
* -----------------------------
* This method returns the type of the expression, which must be one
* of the constants ConstantType, IdentifierType, or CompoundType.
*/
virtual expTypeT getType() = 0;
/*
* Method: eval
* Usage: result = exp->eval(state);
* ---------------------------------
* This method evaluates this expression and returns its value in
* the context of the specified EvalState object.
*/
virtual float eval(EvalState & state) = 0;
/*
* Method: toString
* Usage: str = exp->toString();
* -----------------------------
* This method returns a string representation of this expression.
*/
virtual string toString() = 0;
};
/*
* Class: ConstantNode
* -------------------
* This subclass represents a constant integer expression.
*/
class ConstantNode: public ExpNode {
public:
/*
* Constructor: ConstantNode
* Usage: expressionT exp = new ConstantNode(10);
* ----------------------------------------------
* The constructor initializes a new integer constant expression
* to the given value.
*/
ConstantNode(float val);
/*
* Prototypes for the virtual methods
* ----------------------------------
* These method have the same prototypes as those in the ExpNode
* base class and don't require additional documentation.
*/
virtual expTypeT getType();
virtual float eval(EvalState & state);
virtual std::string toString();
/*
* Method: getValue
* Usage: value = ((ConstantNode *) exp)->getValue();
* --------------------------------------------------
* This method returns the value field without calling eval and
* can be applied only to an object known to be a ConstantNode.
*/
float getValue();
private:
float value;
};
/*
* Class: IdentifierNode
* ---------------------
* This subclass represents a expression corresponding to a variable.
*/
class IdentifierNode : public ExpNode {
public:
/*
* Constructor: IdentifierNode
* Usage: expressionT exp = new IdentifierNode("count");
* -----------------------------------------------------
* The constructor initializes a new identifier expression
* for the variable named by name.
*/
IdentifierNode(std::string name);
/*
* Prototypes for the virtual methods
* ----------------------------------
* These method have the same prototypes as those in the ExpNode
* base class and don't require additional documentation.
*/
virtual expTypeT getType();
virtual float eval(EvalState & state);
virtual std::string toString();
/*
* Method: getName
* Usage: name = ((IdentifierNode *) exp)->getName();
* --------------------------------------------------
* This method returns the name field of the identifier node and
* can be applied only to an object known to be an IdentifierNode.
*/
std::string getName();
private:
std::string name;
};
/*
* Class: CompoundNode
* -------------------
* This subclass represents a compound expression consisting of
* two subexpressions joined by an operator.
*/
class CompoundNode: public ExpNode {
public:
/*
* Constructor: CompoundNode
* Usage: expressionT exp = new CompoundNode('+', e1, e2);
* -------------------------------------------------------
* The constructor initializes a new compound expression
* which is composed of the operator (op) and the left and
* right subexpression (lhs and rhs).
*/
CompoundNode(char op, expressionT lhs, expressionT rhs);
/*
* Destructor: ~CompoundNode
* -------------------------
* The destructor frees any heap storage for this expression.
*/
virtual ~CompoundNode();
/*
* Prototypes for the virtual methods
* ----------------------------------
* These method have the same prototypes as those in the ExpNode
* base class and don't require additional documentation.
*/
virtual expTypeT getType();
virtual float eval(EvalState & state);
virtual std::string toString();
/*
* Methods: getOp, getLHS, getRHS
* Usage: op = ((CompoundNode *) exp)->getOp();
* lhs = ((CompoundNode *) exp)->getLHS();
* rhs = ((CompoundNode *) exp)->getRHS();
* ----------------------------------------------
* These methods return the components of a compound node and can
* be applied only to an object known to be a CompoundNode.
*/
char getOp();
expressionT getLHS();
expressionT getRHS();
private:
char op;
expressionT lhs, rhs;
};
#endif