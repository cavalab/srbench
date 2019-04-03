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
 * Class representing the generalized mean, also known as the power mean, of a
 * set of <code>n</code> numbers. See
 * http://en.wikipedia.org/wiki/Generalized_mean for more discussion.
 * 
 * @author Owen Derby
 */
public class PowerMean extends Mean {

	private final int p;

	/**
	 * @param p the power to use in computing the mean
	 */
	public PowerMean(int p) {
		super();
		this.p = p;
	}

	@Override
	public Double getMean() {
		return Math.pow(sum / (double) n, 1 / (double) p);
	}

	@Override
	public void addValue(Double val) {
		incrementN();
		sum += Math.pow(val, p);
	}

}
