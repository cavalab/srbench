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
package evogpj.math.means;

/**
 * Class defining the basic attributes of a mean class. A mean class is just a
 * light-weight class abstracting the task of computing the mean of a stream of
 * values. Each value is sequentially accumulated as read from the stream, and
 * the running average can be computed at any point. The value computed as the
 * "mean" is defined in the subclasses.
 * 
 * @author Owen Derby
 */
public abstract class Mean {

	protected Double sum;
	protected int n;

	/**
	 * A running mean object, with methods for updating and computing the mean.
	 */
	public Mean() {
            reset();
	}

	/**
	 * Reset the state, as to start computing the mean of a new stream.
	 */
	public void reset() {
		sum = 0.0;
		n = 0;
	}

	/**
	 * @return the current value of the mean
	 */
	public abstract Double getMean();

	/**
	 * Add a new value to the running mean.
	 * 
	 * @param val value to add to the mean.
	 */
	public abstract void addValue(Double val);

	protected void incrementN() {
		n++;
	}

}
