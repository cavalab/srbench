/**
 * Copyright (c) 2011-2013 Evolutionary Design and Optimization Group
 * 
 * Licensed under the MIT License.
 * 
 * See the "LICENSE" file for a copy of the license.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.  
 *
 */
package evogpj.math;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;

/**
 * Abstraction/extension of the genotype evaluation traversal. The problem: we
 * need to traverse the entire genotype tree to evaluate every single training
 * case. This is expensive (in running time). Solution: traverse tree once,
 * capture that traversal as a {@link Function}, and use that for evaluations.
 * That's the purpose of this interface and all implementations of this
 * interface. A further optimization might be to perform some sort of static
 * analysis to intelligently remove useless/dead subexpressions (such as those
 * which always evaluate to the same value, or don't affect any subsequent
 * evaluations).
 * 
 * @author Owen Derby
 */
public abstract class Function {

	/**
	 * Given the set of assignments of features to values for a particular
	 * training case, return the double result of evaluating this function on
	 * that training case.
	 * 
	 * @param t the training case to evaluate.
	 * @return value computed by applying this function to the training case.
	 */
	public abstract Double eval(List<Double> t);
        
        public abstract Double evalIntermediate(List<Double> t, ArrayList<Double> interVals);

	/**
	 * Encapsulate the mapping from a textual label to a Function object's
	 * class. To be used for introspectively determining how to generate a
	 * Function for evaluation.
	 * 
	 * @param label the string from an S-expression encoding a particular
	 *        function
	 * @return the class of the function encoded in label.
	 */
	public static Class<? extends Function> getClassFromLabel(String label) {
		if (label.startsWith("X") || label.equals("x") || label.equals("y")) {
			return Var.class;
		} else if (label.equals("+") || label.equals("plus")) {
			return Plus.class;
		} else if (label.equals("*") || label.equals(".*") || label.equals("times")) {
			return Multiply.class;
		} else if (label.equals("-") || label.equals("minus")) {
			return Minus.class;
		} else if (label.equals("/") || label.equals("./") || label.equals("mydivide")) {
			return Divide.class;
		} else if (label.equals("sin")) {
			return Sin.class;
		} else if (label.equals("cos")) {
			return Cos.class;
		} else if (label.equals("log") || label.equals("mylog")) {
			return Log.class;
		} else if (label.equals("exp")) {
			return Exp.class;
		} else if ((label.equals("sqrt")) || label.equals("mysqrt")) {
			return Sqrt.class;
		} else if (label.equals("square")) {
			return Square.class;
		} else if (label.equals("cube")) {
			return Cube.class;
		} else if (label.equals("quart")) {
			return Quart.class;
		}else{
                    return null;
                }
	}


	/**
	 * Given a label, return the constructor for the class of the function which
	 * represents the label.
	 * 
	 * @param label
	 * @return
	 * @throws SecurityException
	 * @throws NoSuchMethodException
	 */
	public static Constructor<? extends Function> getConstructorFromLabel(String label) throws SecurityException, NoSuchMethodException {
            Class<? extends Function> f = Function.getClassFromLabel(label);
            int arity = Function.getArityFromLabel(label);
            if (arity == 1) {
                return f.getConstructor(Function.class);
            } else if (arity == 2) {
                return f.getConstructor(Function.class, Function.class);
            }
            //return f.getConstructor(String.class);
            return null;
	}

	/**
	 * Simple method for extracting the arity of the function (number of args
	 * the function takes) encoded by the provided label string by introspecting
	 * on the function class represented by the label.
	 * 
	 * @param label string of function, from an S-expression.
	 * @return arity of encoded function
	 */
	public static int getArityFromLabel(String label) {
		Class<? extends Function> f = Function.getClassFromLabel(label);
		if (OneArgFunction.class.isAssignableFrom(f)) {
			return 1;
		} else if (TwoArgFunction.class.isAssignableFrom(f)) {
			return 2;
		} else {
			// Common terminals and default case
			return 0;
		}
	}
	
    /**
     * @return an infix format string representing this function, with #arity %s inclusions.
     */
    public String getInfixFormatString(){
        return "";
    }

    /**
     * @return an infix format string representing this function, with #arity %s inclusions.
     */
    public String getFinalString(){
        return "";
    }
    

}