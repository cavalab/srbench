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
package evogpj.operator;

import evogpj.gp.GPException;
import evogpj.gp.Individual;
import evogpj.gp.Population;

/**
 * Interface specifying generic crossover operation. Crossover is a fundamental
 * operator in genetic programming, producing new individuals for the next
 * generation. The genotypes of two individuals are recombined in some fashion
 * (usually stochastic in nature) to produce one or more (typically two) new
 * individuals.
 * 
 * @author Owen Derby
 */
public interface Crossover {

	/**
	 * Perform crossover between two individuals, returning their offspring. The
	 * two individuals may be thought of as the parents. The returned population
	 * may contain one or more offspring, depending on the crossover
	 * implementation. Neither parents shall be mutated - copies of their genome
	 * should be recombined to form offspring.
	 * 
	 * @param ind1 The first parent
	 * @param ind2 The second parent
	 * @return population The offspring. We return a population because there
	 *         may be more than one offspring returned.
	 * @throws GPException if this operator is not compatible with the genotype
	 *         of one or more of the parents
	 */
	public abstract Population crossOver(Individual ind1, Individual ind2)
			throws GPException;
}
