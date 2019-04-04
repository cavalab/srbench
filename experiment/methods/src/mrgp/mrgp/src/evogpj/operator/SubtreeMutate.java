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
import evogpj.genotype.TreeGenerator;
import evogpj.genotype.TreeNode;
import evogpj.gp.GPException;
import evogpj.gp.Individual;
import evogpj.gp.MersenneTwisterFast;

import java.util.ArrayList;
import java.util.Properties;

import evogpj.algorithm.Parameters;

/**
 * Implement mutation by selecting a node in the tree and generating an entire
 * new subtree, rooted at the selected node.
 * 
 * @author Owen Derby
 */
public class SubtreeMutate extends RandomOperator implements Mutate {

	private final int TREE_MUTATE_MAX_DEPTH;
	private final TreeGenerator treeGen;

    /**
     * Construct a new mutation operator. The maximum depth of the new tree of
     * the returned individual is specified by the value at the key
     * {@value algorithm.Parameters.Names#TREE_MUTATE_MAX_DEPTH} in the
     * properties file. If no value is specified, the default depth limit is
     * {@value algorithm.Parameters.Defaults#TREE_MUTATE_MAX_DEPTH}.
     * 
     * @param rand random number generator.
     * @param props object encoding system properties.
     * @param TGen generator to use for growing new subtrees.
     */
    public SubtreeMutate(MersenneTwisterFast rand, Properties props,TreeGenerator TGen) {
        super(rand);
        if (props.containsKey(Parameters.Names.TREE_MUTATE_MAX_DEPTH))
            TREE_MUTATE_MAX_DEPTH = Integer.valueOf(props.getProperty(Parameters.Names.TREE_MUTATE_MAX_DEPTH));
        else
            TREE_MUTATE_MAX_DEPTH = Parameters.Defaults.TREE_MUTATE_MAX_DEPTH;
        treeGen = TGen;
    }

    @Override
    public Individual mutate(Individual i) throws GPException {
        if (!(i.getGenotype() instanceof Tree)) {
            throw new GPException("attempting SubtreeMutate of genotype not of type Tree");
        }
        Tree copy = (Tree) i.getGenotype().copy();
        ArrayList<TreeNode> treeNodes = copy.getRoot().depthFirstTraversal();
        int whichNode = rand.nextInt(treeNodes.size());
        TreeNode n = treeNodes.get(whichNode);
        // System.out.println("Node picked is: " + whichNode + "; " +
        // n.toString());
        int curDepth = n.getDepth();
        // grow a new subtree at the selected point
        // if selected point is variable
        treeGen.generate(n, TREE_MUTATE_MAX_DEPTH - curDepth, false);
        // System.out.println("New subtree there is: " + n.toStringAsTree());

        return new Individual(copy);
    }

}
