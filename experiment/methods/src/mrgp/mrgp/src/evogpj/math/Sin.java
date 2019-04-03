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
package evogpj.math;

import java.util.ArrayList;
import java.util.List;

public class Sin extends OneArgFunction {

	public Sin(Function op) {
		super(op);
	}

    @Override
    public Double eval(List<Double> t) {
        return Math.sin(arg.eval(t));
    }

    @Override
    public Double evalIntermediate(List<Double> t, ArrayList<Double> interVals) {
        double result = Math.sin(arg.evalIntermediate(t,interVals));
        interVals.add(result);
        return result;
    }
    
    public String getInfixFormatString() {
        //return "sin(%s)";
        return "(sin %s)";
    }
}
