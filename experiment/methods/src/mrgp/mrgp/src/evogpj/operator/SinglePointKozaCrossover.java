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

import evogpj.genotype.Tree;
import evogpj.genotype.TreeNode;
import evogpj.gp.MersenneTwisterFast;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import evogpj.algorithm.Parameters;

/**
 * Extend {@link SinglePointUniformCrossover} as suggested by Koza so that we
 * bias our cross-over point selection. Koza suggests that internal (function)
 * nodes ought to be selected with higher probability than leaf (terminal)
 * nodes. This "promotes the recombining of larger structures," which is more in
 * line with the goals of crossover operation.
 * 
 * @author Owen Derby
 */
public class SinglePointKozaCrossover extends SinglePointUniformCrossover {

	private final double USE_FUNCTION;

	/**
	 * A new mutation operator. The probability for selecting to crossover at an
	 * internal node is specified by the
	 * {@value algorithm.Parameters.Names#KOZA_FUNC_RATE} key in the properties
	 * file. If no value is specified, then the default value of
	 * {@value algorithm.Parameters.Defaults#KOZA_FUNC_RATE} is used.
	 * 
	 * @param rand random number generator.
	 * @param props properties of the system.
	 */
	public SinglePointKozaCrossover(MersenneTwisterFast rand, Properties props) {
		super(rand, props);
		if (props.containsKey(Parameters.Names.KOZA_FUNC_RATE)) {
			USE_FUNCTION = Double.valueOf(props
					.getProperty(Parameters.Names.KOZA_FUNC_RATE));
		} else {
			USE_FUNCTION = Parameters.Defaults.KOZA_FUNC_RATE;
		}
	}

	/**
	 * Implement our biased point selection.
	 */
	@Override
	protected TreeNode selectXOverPt(Tree t) {
		List<TreeNode> nodes = t.getRoot().depthFirstTraversal();
		// Separate nodes into function nodes and terminal nodes.
		List<TreeNode> function_nodes = new ArrayList<TreeNode>();
		List<TreeNode> term_nodes = new ArrayList<TreeNode>();
		for (TreeNode n : nodes) {
			if (n.children.isEmpty())
				term_nodes.add(n);
			else
				function_nodes.add(n);
		}
		// determine which set to select from, then uniformly select from that
		// set.
		if (!function_nodes.isEmpty() && rand.nextDouble() <= USE_FUNCTION)
			return function_nodes.get(rand.nextInt(function_nodes.size()));
		else
			return term_nodes.get(rand.nextInt(term_nodes.size()));
	}

}
