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

import evogpj.gp.Individual;
import evogpj.gp.Population;

/**
 * Select Operation. The select operation must take a Population and return an
 * Individual from the population. How that Individual is chosen is left up to
 * the operator design.
 * 
 * @author Owen Derby
 */
public interface Select {

	/**
	 * Given a population, select an individual and return it.
	 * 
	 * @param pop the population to select an individual from.
	 * @return the selected individual.
	 */
	public abstract Individual select(Population pop);
}
