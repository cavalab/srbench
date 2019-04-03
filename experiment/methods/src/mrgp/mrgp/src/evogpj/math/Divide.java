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

public class Divide extends TwoArgFunction {

	public Divide(Function a1, Function a2) {
		super(a1, a2);
	}

    @Override
    public Double eval(List<Double> t) {
        Double denom = arg2.eval(t);
        if (Math.abs(denom) < 1e-6) {
                return (double) 1; // cc Silva 2008 thesis
        } else {
                return arg1.eval(t) / denom;
        }
    }

    @Override
    public Double evalIntermediate(List<Double> t, ArrayList<Double> interVals) {
        double result;
        Double denom = arg2.evalIntermediate(t,interVals);
        if (Math.abs(denom) < 1e-6) {
                result = (double) 1; // cc Silva 2008 thesis
        } else {
                result = arg1.evalIntermediate(t,interVals) / denom;
        }
        interVals.add(result);
        return result;
    }
    public String getInfixFormatString() {
        //return "mydivide(%s, %s)";
        return "(mydivide %s %s)";
    }
}
