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
package evogpj.genotype;

import evogpj.gp.GPException;
import java.util.List;
import evogpj.math.Function;

/**
 * Class which represents the genotype of an individual as a rooted tree. The
 * actual genotype is actually comprised of a tree of many TreeNode instances.
 * Because the particular contents (nodes) of the genotype (tree) will change
 * (crossover, mutation), we need an immutable handle to this mutating genotype.
 * We achieve this by keeping a private TreeNode handler, with no other purpose
 * than to point to the root of the tree. So this class boils down to a fancy
 * wrapper around this immutable handle node, but represents the abstract
 * concept of the Tree genotype.
 * 
 * @author Owen Derby
 * @see TreeNode
 */
public class Tree extends Genotype {
    private static final long serialVersionUID = -3871767863867101731L;

    // A TreeNode with the root of the Tree as its only child (effectively a
    // pointer to the root of the tree). This level of indirection allows the
    // entire tree to change (meaning we can swap root nodes) without creating a
    // new Tree instance.
    private TreeNode holder;
    private Integer subtreeComplexity;
    // the coefficients for scaling the models via linear regression
    private Double[] scalingCoeffs;

    public Tree(TreeNode aHolder) {
        this.holder = aHolder;
        this.subtreeComplexity = null;
        this.scalingCoeffs = new Double[2];
    }

    @Override
    public Genotype copy() {
        // hehe, the old serialise-deserialise trick
        Tree cp = TreeGenerator.generateTree(this.toPrefixString());
        cp.subtreeComplexity = this.subtreeComplexity;
        return cp;
    }

    public TreeNode getRoot() {
        return holder.children.get(0);
    }

    /**
     * Default: return unscaled MATLAB infix string
     * @return 
     */
    @Override
    public String toString() {
        //return toInfixString();
        return toPrefixString();
    }

    public String toPrefixString() {
        return getRoot().toStringAsTree();
    }
    
    public String toFinalPrefixString() {
        return getRoot().toFinalStringAsTree();
    }

    public String toScaledString() {
        double slope = this.getScalingSlope();
        double intercept = this.getScalingIntercept();
        return String.format("(%.16f .* %s) + %.16f", slope, this.toPrefixString(), intercept);
    }
    
    public String toFinalScaledString() {
        double slope = this.getScalingSlope();
        double intercept = this.getScalingIntercept();
        return String.format("(%.16f .* %s) + %.16f", slope, this.toFinalPrefixString(), intercept);
    }
        
    public boolean evalBoolean(List<Boolean> t) {
        return getRoot().evalBoolean(t);
    }

    @Override
    public Boolean equals(Genotype other) {
        if (!(other.getClass() == this.getClass()))
                return false;
        Tree otherTree = (Tree) other;
        boolean equal = this.equals(otherTree);
        return equal;
    }

    /**
     * String comparison as equals
     * @param otherTree
     * @return
     */
    public Boolean equals(Tree otherTree) {
        String thisString = this.toPrefixString();
        String otherString = otherTree.toPrefixString();
        boolean equal = thisString.equals(otherString);
        return equal;
    }

    /**
     * Compute the size (number of nodes) of the Tree.
     * 
     * @return integer size
     */
    public int getSize() {
        return getRoot().getSubtreeSize();
    }

    /**
     * Compute the depth of the Tree. The depth is the longest simple path from
     * the root to a terminal node.
     * 
     * @return integer depth
     */
    public int getDepth() {
        return getRoot().getSubtreeDepth();
    }

    /**
     * @return the "a" coefficient in y = ax + b
     */
    public Double getScalingSlope() {
        return scalingCoeffs[0];
    }

    /**
     * Set the scaling slope (the "b" in y = ax + b) of this individual
     */
    public void setScalingSlope(Double slope) {
        scalingCoeffs[0] = slope;
    }

    /**
     * @return the "b" coefficient in y = ax + b
     */
    public Double getScalingIntercept() {
        return scalingCoeffs[1];
    }

    /**
     * Set the scaling intercept (the "b" in y = ax + b) of this individual 
     */
    public void setScalingIntercept(Double intercept) {
        scalingCoeffs[1] = intercept;
    }

    /**
     * Create a new {@link Function} object to represent this tree in
     * preparation for computing the fitness.
     * 
     * @return
     * @see Function
     */
    public Function generate() {
        try {
            TreeNode r = getRoot();
            Function f = r.generate();
            return f;
        } catch (GPException e) {
            System.exit(-1);
        }
        return null;
    }
    
    /**
     *
     * A memorized method which computes the complexity of an individual
     * by summing the sizes of all subtrees. Referenced in "Model-based
     * Problem Solving through Symbolic Regression via Pareto Genetic
     * Programming", Vladislavleva 2008
     */
    // TODO add back in smart memoization which handles recalculation
    // upon crossover
    public int getSubtreeComplexity() {
        subtreeComplexity = this.getRoot().getSubtreeComplexity();
        return subtreeComplexity;
    }
}
