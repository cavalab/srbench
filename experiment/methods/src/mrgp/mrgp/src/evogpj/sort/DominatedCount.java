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
package evogpj.sort;

import evogpj.evaluation.FitnessComparisonStandardizer;
import evogpj.evaluation.FitnessFunction;
import evogpj.gp.Individual;
import evogpj.gp.Population;

import java.util.LinkedHashMap;
import java.util.Set;

import evogpj.operator.Operator;

/**
 * A class for performing a non-dominated sort of individuals
 * 
 * @author Dylan Sherry and Ignacio Arnaldo
 */
public class DominatedCount extends Operator {

    /**
     * Performs a O(mn^2) non-dominated sort, where m is the number of objective
     * functions and n is the population size
     * 
     * @param p
     * @param f
     * @throws DominationException
     */
    public static void countDominated(Population p,LinkedHashMap<String, FitnessFunction> f) throws DominationException {
        for (int i = 0; i < p.size(); i++) {		
            Individual a = p.get(i);
            for (int j = i + 1; j < p.size(); j++) {
                Individual b = p.get(j);
                Boolean aDomb = domination(a, b, f);
                Boolean bDoma = domination(b, a, f);
                if (!aDomb && !bDoma && a.getFitnesses().equals(b.getFitnesses())) { // happens if a and b are identical
                    // in this case whoever happens to be located earlier in the population wins
                    if (i <= j) b.incrementDominationCount();
                    else a.incrementDominationCount();
                } else if (aDomb) {
                    b.incrementDominationCount();
                } else if (bDoma) {
                    a.incrementDominationCount();
                }
            }
        }
    }

    public static class DominationException extends Exception {
        private static final long serialVersionUID = 243416542135464L;
        public DominationException(String s) {super(s);}
    }

    /**
     * The domination operator returns true if A dominates B, which occurs when
     * (1) A is at least as good as B in all objectives, and (2) A is strictly
     * better than B in at least one objective
     * 
     * @param a
     * @param b
     * @param fitnessFunctions
     * @return 
     * @throws evogpj.sort.DominatedCount.DominationException
     */
    @SuppressWarnings("static-access")
    public static boolean domination(Individual a, Individual b, LinkedHashMap<String, FitnessFunction> fitnessFunctions )throws DominationException {
        Set<String> aFuncNames = a.getFitnessNames();
        Set<String> bFuncNames = b.getFitnessNames();
        if (aFuncNames.size() != bFuncNames.size())
            throw new DominationException(("Error: individuals' fitnesses are not of same " + "length: a=\"%s\" and b=\"%s\"").format(a.toString(),b.toString()));
        Boolean strictlyBetter = false;
        for (String fitnessFunctionName : aFuncNames) {
            Double af = FitnessComparisonStandardizer.getFitnessForMinimization(a, fitnessFunctionName, fitnessFunctions);
            Double bf = FitnessComparisonStandardizer.getFitnessForMinimization(b, fitnessFunctionName, fitnessFunctions);
            if (!(af <= bf)) // domination violated
                return false;
            if (af < bf)
                strictlyBetter = true;
        }
        return strictlyBetter;
    }
}
