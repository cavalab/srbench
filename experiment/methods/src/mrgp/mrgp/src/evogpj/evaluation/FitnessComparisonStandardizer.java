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

import evogpj.gp.Individual;
import java.util.LinkedHashMap;

/**
 * This class standardizes the way fitness values are compared. If a particular
 * fitness function is a maximizing function (i.e. fitness on a data set, which
 * here is an inversion of mean squared error), this class will numerically
 * invert it, which means it can now be treated as a fitness value from a
 * minimization problem, which is considered the default type of objective
 * function in evogpj. This simplifies any logic involving the comparison of two
 * individuals' fitness values for a given function.
 * 
 * @author Dylan
 * 
 */
public class FitnessComparisonStandardizer {

	/**
	 * Standardizes the way by which fitnesses are compared. Will always return
	 * a fitness score where lower indicates better.
	 * 
	 * @return
	 */
	public static Double getFitnessForMinimization(Individual individual,String funcName, LinkedHashMap<String, FitnessFunction> fitnessFunctions) {
		Double fitness = individual.getFitness(funcName);
		if (fitnessFunctions.get(funcName).isMaximizingFunction()) {
			fitness = invert(fitness);
		}
		return fitness;
	}
        
	/**
	 * Standardizes the way by which fitnesses are compared. Will always return
	 * a fitness score where lower indicates better.
	 * 
	 * @param fitnesses a LinkedHashMap mapping fitness function name to fitness score
	 * @return
	 */
	public static Double getFitnessForMinimization(LinkedHashMap<String, Double> fitnesses, String funcName, LinkedHashMap<String, FitnessFunction> fitnessFunctions) {
            Double fitness = fitnesses.get(funcName);
            return handleInversionForMinimization(fitness, fitnessFunctions.get(funcName));
	}
        
        private static Double handleInversionForMinimization(Double fitnessValue, FitnessFunction fitnessFunction) {
            if (fitnessFunction.isMaximizingFunction())
                fitnessValue = invert(fitnessValue);
            return fitnessValue;
	}
	/**
	 * Invert a fitness value, thus converting from maximization to minimization
	 * score, or vice versa
	 * 
	 * @param input
	 * @return
	 */
	public static Double invert(Double fitness) {
		return (1 - fitness) / (1 + fitness);
	}
}
