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

import evogpj.genotype.TreeGenerator;
import evogpj.gp.Individual;
import evogpj.gp.MersenneTwisterFast;
import evogpj.gp.Population;

import java.util.Properties;

import evogpj.algorithm.Parameters;
import evogpj.genotype.Tree;

/**
 * Create new population of Trees using Koza's ramped half and half
 * initialization.
 * 
 * @author Owen Derby
 */
public class TreeInitialize extends RandomOperator implements Initialize {
	private final int TREE_INITIAL_MAX_DEPTH;
	private final TreeGenerator treeGen;

	/**
	 * Create new intialize operator. The generated trees are limited by a
	 * maximum allowed depth. This limit is specified by the value at the key
	 * {@value algorithm.Parameters.Names#TREE_MUTATE_MAX_DEPTH} in the
	 * properties file. If there is no value specified, the default limit of
	 * {@value algorithm.Parameters.Defaults#TREE_INIT_MAX_DEPTH} is used.
	 * 
	 * @param rand random number generator instance.
	 * @param props properties file.
	 * @param TGen generator for creating the new trees.
	 */
	public TreeInitialize(MersenneTwisterFast rand, Properties props,TreeGenerator TGen) {
		super(rand);
		if (props.containsKey(Parameters.Names.TREE_INIT_MAX_DEPTH))
                    TREE_INITIAL_MAX_DEPTH = Integer.valueOf(props.getProperty(Parameters.Names.TREE_INIT_MAX_DEPTH));
		else
                    TREE_INITIAL_MAX_DEPTH = Parameters.Defaults.TREE_INIT_MAX_DEPTH;
		this.treeGen = TGen;
	}

	@Override
	public Population initialize(int popSize) {
		Population ret = new Population();
		// number of individuals to generate per depth, per type (grow or full)
		int inds_per_d = popSize / 2 / TREE_INITIAL_MAX_DEPTH;

		for (int depth = 1; depth <= TREE_INITIAL_MAX_DEPTH; depth++) {
			for (int i = 0; i < inds_per_d; i++) {
                            Tree t = treeGen.generateTree(depth, true);
                            Individual ind = new Individual(t);
                            ret.add(ind);
                            //ret.add(new Individual(treeGen.generateTree(depth, true)));
                            ret.add(new Individual(treeGen.generateTree(depth, false)));
			}
		}
		return ret;
	}

}
