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

import evogpj.gp.Individual;

import java.io.Serializable;


/**
 * A genotype represents the underlying solution represented by an
 * {@link Individual}. Akin to it's namesake in Biology, the genotype tells us
 * how the individual operates (as opposed to the {@link Phenotype}, which tells
 * us how the individual appears). In Koza's terminology, it is the candidate
 * program of the individual.
 * 
 * @author Owen Derby
 * @see Individual
 * @see Phenotype
 */
public abstract class Genotype implements Serializable {
	private static final long serialVersionUID = 2834690745385096387L;

	/**
	 * Perform a deep copy of the genotype
	 * 
	 * @return new genotype copy
	 */
	public abstract Genotype copy();

	public abstract Boolean equals(Genotype other);
}
