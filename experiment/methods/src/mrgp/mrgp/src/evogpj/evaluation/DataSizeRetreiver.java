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
package evogpj.evaluation;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;

/**
 * A class for crudely determining the number of lines and explanatory variables
 * in a data file
 * 
 * @author Dylan Sherry
 * 
 */
public class DataSizeRetreiver {
	/**
	 * Set the number of terminals present in the file
	 * 
	 * @throws IOException
	 */
	public static int num_terminals(String filepath) {
		try {
			BufferedReader f = new BufferedReader(new FileReader(filepath));
			if (!f.ready()) {
				System.err.println(String.format("Error: empty file %s", filepath));
			}
			//String[] tokens = f.readLine().split("\\t");
                        String[] tokens = f.readLine().split(",");
			f.close();
			return tokens.length - 1; // assuming one output variable
		} catch (FileNotFoundException e) {
			System.err.println(String.format("Error: file %s not found.", filepath));
			System.exit(-1);
		} catch (IOException e) {
			System.exit(-1);
		}
		return -1;
	}

	/**
	 * Set the number of lines found in the file
	 * 
	 * @throws IOException
	 */
	public static int num_fitness_cases(String filepath) {
		try {
			InputStream is = new BufferedInputStream(new FileInputStream(filepath));
			try {
				byte[] c = new byte[1024];
				int count = 0;
				int readChars;
				boolean empty = true;
				while ((readChars = is.read(c)) != -1) {
					empty = false;
					for (int i = 0; i < readChars; ++i) {
						if (c[i] == '\n') {
							++count;
						}
					}
				}
				return (count == 0 && !empty) ? 1 : count;
			} finally {
				is.close();
			}
		} catch (FileNotFoundException e) {
			System.err.println(String.format("Error: file %s not found.", filepath));
			System.exit(-1);
		} catch (IOException e) {
			System.exit(-1);
		}
		return -1;
	}
}
