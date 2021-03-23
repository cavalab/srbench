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

/**
 * Outline how to interface with underlying data. We view data as have to
 * distinct components. The first is the rows of input data, the <i>input</i>.
 * The second is the output value for each row of input, the <i>target</i>.
 * Further, we have a concept of <i>scaled</i> target values. See
 * {@link SRFitness} for how we use this.
 * 
 * @author Owen Derby
 */
public interface DataJava {
	/**
	 * Get the current set of input data, where each row of input corresponds to
	 * a list of values in the return.
	 * 
	 * @return Two Dimensional Array; equivalent to the matrix of
	 *         input data, where each training case corresponds to a row.
	 */
	public double[][] getInputValues();

	/**
	 * Get the target values corresponding to the training cases. The ith targt
	 * value is the output corresponding to the ith row of the input values (the
	 * ith training case).
	 * 
	 * @return One Dimensional Array; equivalent to the column of output
	 *         values, where each value is the output for the given row of
	 *         input.
	 */
	public double[] getTargetValues();

	/**
	 * Returns the precomputed mean on the target values. Used in the ModelScaler
	 */
	public Double getTargetMean();
	
	/**
	 * Return the maximum of all the target values.
	 * @return
	 */
	public Double getTargetMax();

	/**
	 * Return the minimum of all target values.
	 * @return
	 */
	public Double getTargetMin();

	/**
	 * Get the target values, scaled to be in the range [0,1]. This is done
	 * according to the following formula.
	 * <p>
	 * <code>y_i_scaled = (y_i-getTargetMin())/(getTargetMax()-getTargetMin())</code>
	 * <p>
	 * Otherwise, this has the same properties as {@link #getTargetValues()}.
	 * 
	 * @return
	 */
	public double[] getScaledTargetValues();
        
        /**
         * @return the number Of Fitness Cases
         */
        public int getNumberOfFitnessCases();

        /**
         * @return the number Of Features
         */
        public int getNumberOfFeatures();
}
