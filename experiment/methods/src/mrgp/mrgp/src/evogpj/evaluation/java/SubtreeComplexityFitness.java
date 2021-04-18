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
package evogpj.evaluation.java;

import evogpj.algorithm.Parameters;
import evogpj.evaluation.FitnessFunction;
import evogpj.genotype.Tree;
import evogpj.gp.Individual;
import evogpj.gp.Population;

/**
 * Evaluates fitness by calculating an individual's subtree complexity
 *
 * @author Dylan Sherry
 */
public class SubtreeComplexityFitness extends FitnessFunction {

	public static final String FITNESS_KEY = Parameters.Operators.SUBTREE_COMPLEXITY_FITNESS;

	public Boolean isMaximizingFunction = false;

	// this function provides discrete fitness values (integer)
	public Boolean discreteFitness = true;

	/**
	 * Should this fitness function be minimized (i.e. mean squared error) or
	 * maximized?
     * @return 
	 */
        @Override
	public Boolean isMaximizingFunction() {
		return this.isMaximizingFunction;
	}

	/**
	 * Compute the subtree complexity of the individual
     * @param ind
	 */
	public void eval(Individual ind) {
		Tree t = (Tree) ind.getGenotype();
		Integer complexity = t.getSubtreeComplexity();
		ind.setFitness(SubtreeComplexityFitness.FITNESS_KEY, (double) complexity);
	}

	@Override
	public void evalPop(Population pop) {
		for (Individual individual : pop) {
			this.eval(individual);
		}
	}
}
