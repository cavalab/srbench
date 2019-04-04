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
package evogpj.evaluation;

import evogpj.gp.Population;

/**
 * Base class for all fitness evaluators. Fitness functions are fundamentally
 * different from the other operators in the operators package because they need
 * to interact with both genotypes and phenotypes of individuals and do not
 * manipulate the genotype of individuals.
 * 
 * @author Owen Derby
 */
public abstract class FitnessFunction {

	// used to access this fitness function's position in the
	// gp.Individual.fitnesses HashMap. Each subclass must
	// define this value
	//public static String FITNESS_KEY;

	// should this value be maximized (true) or minimized (false)?
	// used by non-dominated sort to convert all fitness functions
	// to minimizers
	// public Boolean isMaximizingFunction;

	/**
	 * Evaluate a single individual
	 * 
	 * @param pop
	 */

	/**
	 * Evaluate each individual in a population
	 * 
	 * @param pop
	 */
	public abstract void evalPop(Population pop);

	public abstract Boolean isMaximizingFunction();

}
