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

import java.util.List;
import java.util.Properties;

import evogpj.algorithm.Parameters;

import evogpj.genotype.Tree;
import evogpj.genotype.TreeNode;
import evogpj.gp.GPException;
import evogpj.gp.Individual;
import evogpj.gp.MersenneTwisterFast;
import evogpj.gp.Population;

/**
 * Recombine two individuals by swapping subtrees. A single point (node) is
 * selected, uniformly at random, in each individual (the parents) and then the
 * two subtrees rooted at those nodes are swapped, resulting in two new
 * individuals (the offspring). This operator will only work on individuals with
 * genotypes of type Tree. Further, there is an upper bound on the maximum
 * allowable depth for the offspring. Crossover is attempted repeatedly until
 * both offspring are of allowable depth or a maximum number of tries are
 * attempted.
 * 
 * @author Owen Derby
 */
public class SinglePointUniformCrossover extends RandomOperator implements Crossover {

    private final int TREE_XOVER_MAX_DEPTH;
    private final int TREE_XOVER_TRIES;

    /**
     * Create crossover operator which swaps subtrees at uniformly selected
     * points in two parent individuals. There are two parameters for this
     * operator.
     * <ul>
     * <li>The maximum depth allowed in new trees, specified by the key
     * {@value algorithm.Parameters.Names#TREE_XOVER_MAX_DEPTH}, which defaults
     * to {@value algorithm.Parameters.Defaults#TREE_XOVER_MAX_DEPTH}.
     * <li>The maximum number of times to try the crossover before giving up. A
     * crossover attempt fails if either of the new trees is too deep. This
     * parameter is specified by the key
     * {@value algorithm.Parameters.Names#TREE_XOVER_TRIES}, which defaults to
     * {@value algorithm.Parameters.Defaults#TREE_XOVER_TRIES}.
     * 
     * @param rand
     * @param props
     */
    public SinglePointUniformCrossover(MersenneTwisterFast rand,Properties props) {
        super(rand);
        if (props.containsKey(Parameters.Names.TREE_XOVER_MAX_DEPTH))
            TREE_XOVER_MAX_DEPTH = Integer.valueOf(props.getProperty(Parameters.Names.TREE_XOVER_MAX_DEPTH));
        else
            TREE_XOVER_MAX_DEPTH = Parameters.Defaults.TREE_XOVER_MAX_DEPTH;
        if (props.containsKey(Parameters.Names.TREE_XOVER_TRIES))
            TREE_XOVER_TRIES = Integer.valueOf(props.getProperty(Parameters.Names.TREE_XOVER_TRIES));
        else
            TREE_XOVER_TRIES = Parameters.Defaults.TREE_XOVER_TRIES;
    }

    @Override
    public Population crossOver(Individual ind1, Individual ind2) throws GPException {
            if (!(ind1.getGenotype() instanceof Tree && ind2.getGenotype() instanceof Tree)) {
                throw new GPException("attempting SinglePointUniformCrossover of two genotypes not of type Tree");
            }
		Tree c1, c2;
		int tries = 0;
		do {
			c2 = (Tree) ind2.getGenotype().copy();
			c1 = (Tree) ind1.getGenotype().copy();
			// pick a xover pt in this Tree by uniform sampling of
			// depthFirstTraversal, and find that node's index among its
			// siblings
			TreeNode xoverPt1 = selectXOverPt(c1);
			int xoverPt1idxInChildren = xoverPt1.parent.children
					.indexOf(xoverPt1);

			// same for other Tree
			TreeNode xoverPt2 = selectXOverPt(c2);
			int xoverPt2idxInChildren = xoverPt2.parent.children
					.indexOf(xoverPt2);

			// other.xoverpt = this.xoverpt, and fix up parent link
			xoverPt2.parent.children.set(xoverPt2idxInChildren, xoverPt1);
			TreeNode tmpParent = xoverPt1.parent;
			xoverPt1.parent = xoverPt2.parent;

			// this.xoverpt = other.xoverpt, and fix up parent link
			tmpParent.children.set(xoverPt1idxInChildren, xoverPt2);
			xoverPt2.parent = tmpParent;

			// reset cached values
			xoverPt1.reset();
			xoverPt2.reset();
			tries++;
		} while ((c1.getDepth() > TREE_XOVER_MAX_DEPTH || c2.getDepth() > TREE_XOVER_MAX_DEPTH) && tries < TREE_XOVER_TRIES);
		Population twoPop = new Population();
		if (tries >= TREE_XOVER_TRIES) {
			// System.out.println("failed to xover properly"+ c1.getDepth() +
			// " " +c2.getDepth());

			// One of the children might be of permissible depth...
			if (c1.getDepth() > TREE_XOVER_MAX_DEPTH)
				twoPop.add(ind1.copy());
			else
				twoPop.add(new Individual(c1));

			if (c1.getDepth() > TREE_XOVER_MAX_DEPTH)
				twoPop.add(ind2.copy());
			else
				twoPop.add(new Individual(c2));
		} else {
			Individual i1 = new Individual(c1);
			Individual i2 = new Individual(c2);
			// clear any nonessential memoized values
			i1.reset();
			i2.reset();
			twoPop.add(i1);
			twoPop.add(i2);
		}
		return twoPop;
	}

	/**
	 * Select point (node) uniformly in the given tree.
	 * 
	 * @param t Tree to select from
	 * @return Chosen node
	 */
	protected TreeNode selectXOverPt(Tree t) {
		List<TreeNode> nodes = t.getRoot().depthFirstTraversal();
		int nodeCount = nodes.size();
		int xoverPt1idx = rand.nextInt(nodeCount);
		return nodes.get(xoverPt1idx);
	}

}
