package edu.uci.lasso;

public class LassoFit {
        // Number of lambda values
        public int numberOfLambdas;

        // Intercepts
        public double[] intercepts;

        // Compressed weights for each solution
        public double[][] compressedWeights;

        // Pointers to compressed weights
        public int[] indices;

        // Number of weights for each solution
        public int[] numberOfWeights;

        // Number of non-zero weights for each solution
        public int[] nonZeroWeights;

        // The value of lambdas for each solution
        public double[] lambdas;

        // R^2 value for each solution
        public double[] rsquared;

        // Total number of passes over data
        public int numberOfPasses;

        private int numFeatures;

        public LassoFit(int numberOfLambdas, int maxAllowedFeaturesAlongPath, int numFeatures) {
                intercepts = new double[numberOfLambdas];
                compressedWeights = MathUtil.allocateDoubleMatrix(numberOfLambdas, maxAllowedFeaturesAlongPath);
                indices = new int[maxAllowedFeaturesAlongPath];
                numberOfWeights = new int[numberOfLambdas];
                lambdas = new double[numberOfLambdas];
                rsquared = new double[numberOfLambdas];
                nonZeroWeights = new int[numberOfLambdas];
                this.numFeatures = numFeatures;
        }

        public double[] getWeights(int lambdaIdx) {
                double[] weights = new double[numFeatures];
                for (int i = 0; i < numberOfWeights[lambdaIdx]; i++) {
                        weights[indices[i]] = compressedWeights[lambdaIdx][i];
                }
                return weights;
        }

}
