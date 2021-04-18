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
import evogpj.gp.MersenneTwisterFast;

import java.util.List;
import java.util.StringTokenizer;

import evogpj.math.Function;
import evogpj.algorithm.Parameters;
import java.util.ArrayList;

/**
 * Factory class for creating Trees.
 * 
 * @author Owen Derby
 * @see Tree
 */
public class TreeGenerator {

    private MersenneTwisterFast rng;
    private final List<String> terminals;
    private final List<String> functions;

    /**
     * Create the factory, setting the terminal and function sets to use. If no
     * function sets are provided, the defaults are used from
     * {@link algorithm.Parameters.Defaults}.
     * 
     * @param r An instance of {@link MersenneTwisterFast} to use when selecting
     *        labels for nodes in trees.
     * @param funcset The set of functions to select internal node labels from.
     * @param termset The set of terminals to select terminal node labels from.
     */
    public TreeGenerator(MersenneTwisterFast r, List<String> funcset,List<String> termset) {
        this.rng = r;
        if (funcset == null) {
            System.out.println("No function set provided - using defaults!");
            functions = Parameters.Defaults.FUNCTIONS;
        } else {
            functions = funcset;
        }
        if (termset == null) {
            System.out.println("No terminal set provided - using defaults!");
            terminals = Parameters.Defaults.TERMINALS;
        } else {
            terminals = termset;
        }
    }

    public Tree generateLinearModel(List<String> termset) {
        ArrayList<String> featureList = new ArrayList<String>();
        featureList.addAll(termset);
        TreeNode holder = new TreeNode(null, "holder");
        TreeNode root = new TreeNode(holder, "null");
        generateLinearModel(root, featureList);
        holder.addChild(root);
        Tree t = new Tree(holder);
        return t;
    }
    
    public void generateLinearModel(TreeNode node,List<String> featureList) {
        // See node as a placeholder: we overwrite its label,
        // children, but we don't change its parent or id.
        node.children.clear();
        if (featureList.size() == 1) {
            // can't go deeper, make it a terminal
            node.label = featureList.get(0);
        } else {
            node.label = "+";
            TreeNode leftNode = new TreeNode(node, "null");
            int indexFeature = rng.nextInt(featureList.size());
            leftNode.label = featureList.get(indexFeature);
            featureList.remove(indexFeature);
            node.addChild(leftNode);
            
            TreeNode rightNode = new TreeNode(node, "null");
            generateLinearModel(rightNode, featureList);
            node.addChild(rightNode);
        }
        node.resetAbove();
    }
    
    /**
     * Generate a tree by either the grow or the fill method (as described by
     * Koza). If using the fill method, the new tree will be a full depth with
     * every terminal at maxDepth. If using the grow method, then the tree is
     * "grown" by randomly choosing either a terminal or function for each node,
     * and recursing for each function (non-terminal) selected. This continues
     * until all terminals are selected or maxDepth is reached.
     * 
     * @param maxDepth the maximum depth the new tree can reach.
     * @param full Boolean value indicating to use fill or grow method. If true,
     *        the fill method will be used. If false, the grow method.
     * @return the newly generated instance of the tree.
     */
    public Tree generateTree(int maxDepth, boolean full) {
        TreeNode holder = new TreeNode(null, "holder");
        TreeNode root = new TreeNode(holder, "null");
        generate(root, maxDepth, full);
        holder.addChild(root);
        Tree t = new Tree(holder);
        return t;
    }

    /**
     * Constructor a tree from a string. The string is assumed to be a LISP-like
     * S-expression, using prefix notation for operators.
     * 
     * @param input The S-expression string.
     * @return new tree representing the inpu.
     */
    public static Tree generateTree(String input) {
        // Make sure the string is tokenizable
        // FIXME allow other delimiters?
        input = input.replace("(", " ( ");
        input = input.replace("[", " [ ");
        input = input.replace(")", " ) ");
        input = input.replace("]", " ] ");

        StringTokenizer st = new StringTokenizer(input);
        TreeNode holder = new TreeNode(null, "holder");
        parseString(holder, st);
        return new Tree(holder);
    }
    
    /**
     * Constructor a tree from a string. The string is assumed to be a LISP-like
     * S-expression, using prefix notation for operators.
     * 
     * @param input The S-expression string.
     * @return new tree representing the inpu.
     */
    public static Tree generateTreeConstants(String input) {
        // Make sure the string is tokenizable
        // FIXME allow other delimiters?
        input = input.replace("(", " ( ");
        input = input.replace("[", " [ ");
        input = input.replace(")", " ) ");
        input = input.replace("]", " ] ");

        StringTokenizer st = new StringTokenizer(input);
        TreeNode holder = new TreeNode(null, "holder");
        parseString(holder, st);
        return new Tree(holder);
    }

    /**
     * Given a node in a tree, generate an entire new subtree below, using
     * either the fill or grow method (see {@link #generateTree(int, boolean)}).
     * Note that the given node will have it's label overwritten and all it's
     * current children cleared.
     * 
     * @param node TreeNode to use as the root of the new subtree.
     * @param depth The allowed depth of the generated subtree.
     * @param full Whether to use the fill or grow method for generating the
     *        subtree.
     */
    public void generate(TreeNode node, int depth, boolean full) {
        // See node as a placeholder: we overwrite its label,
        // children, but we don't change its parent or id.
        node.children.clear();
        if (depth <= 0) {
            // can't go deeper, make it a terminal
            node.label = terminals.get(rng.nextInt(terminals.size()));
        } else {
            int label_idx;
            if (full) {
                // want full tree, so make it a function
                label_idx = rng.nextInt(functions.size());
            } else {
                // growing tree, randomly choose function or terminal
                label_idx = rng.nextInt(functions.size() + terminals.size());
            }
            if (label_idx < functions.size()) {
                node.label = functions.get(label_idx);
                // create and recurse on each argument (child) of function
                for (int i = 0; i < arity(node.label); i++) {
                    TreeNode newNode = new TreeNode(node, "null");
                    node.addChild(newNode);
                    generate(newNode, depth - 1, full);
                }
            } else {
                node.label = terminals.get(label_idx - functions.size());
            }
        }
        node.resetAbove();
    }
    
    /**
     * Given a node in a tree, generate an entire new subtree below, using
     * either the fill or grow method (see {@link #generateTree(int, boolean)}).
     * Note that the given node will have it's label overwritten and all it's
     * current children cleared.
     * 
     * @param node TreeNode to use as the root of the new subtree.
     * @param depth The allowed depth of the generated subtree.
     * @param full Whether to use the fill or grow method for generating the
     *        subtree.
     */
    public void generateForMutationWithConstants(TreeNode node, int depth, boolean full) throws GPException {
        // See node as a placeholder: we overwrite its label,
        // children, but we don't change its parent or id.
        
        if (node.children.isEmpty()) {
            node.children.clear();
            if(rng.nextBoolean()){// swap variable
                node.label = terminals.get(rng.nextInt(terminals.size()));
                node.resetCoeff();
            }else{ // change constant
                if(rng.nextBoolean()){// increase coefficient a 10%
                    node.increaseCoeff();
                }else{// decrease coefficient a 10%
                    node.decreaseCoeff();
                }
                
            }
        } else {
            generate(node, depth, full);
        }
        node.resetAbove();
    }
    
    public void generateChangeWeights(TreeNode node, int depth, boolean full) throws GPException {
        node.children.clear();
        if(rng.nextBoolean()){// increase coefficient a 10%
            node.increaseCoeff();
        }else{// decrease coefficient a 10%
            node.decreaseCoeff();
        }
        node.resetAbove();
    }

	/**
	 * Compute the arity of the function represented by the string.
	 * 
	 * @param label string encoding a function (ie "sin" or "*")
	 * @return the arity of the function (number of arguments the function
	 *         takes)
	 * @see math.Function#getArityFromLabel(String)
	 */
	public static int arity(String label) {
		// Boolean problems functions:
		if (label.equals("if")) {
			return 3;
		} else if (label.equals("and")) {
			return 2;
		} else if (label.equals("or")) {
			return 2;
		} else if (label.equals("nand")) {
			return 2;
		} else if (label.equals("nor")) {
			return 2;
		} else if (label.equals("not")) {
			return 1;
		}
		// Symbolic Regression functions
		return Function.getArityFromLabel(label);
	}

    private static void parseString(TreeNode parent, StringTokenizer st) {

        while (st.hasMoreTokens()) {
            String currTok = st.nextToken().trim();

            if (currTok.equals("")) {                    
            } else if (currTok.equals("(") || currTok.equals("[")) {
                // The next token is the parent of a new subtree
                currTok = st.nextToken().trim();
                TreeNode newNode = new TreeNode(parent, currTok);
                parent.addChild(newNode);
                parseString(newNode, st);
            } else if (currTok.equals(")") || currTok.equals("]")) {
                // Finish this subtree
                return;
            } else {
                // An ordinary child node: add it to parent and continue.
                TreeNode newNode = new TreeNode(parent, currTok);
                parent.addChild(newNode);
            }
        }
    }

}
