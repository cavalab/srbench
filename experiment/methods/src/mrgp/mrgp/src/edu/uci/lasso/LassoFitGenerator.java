package edu.uci.lasso;

/**
 * This implemenation is based on: Friedman, J., Hastie, T. and Tibshirani, R.
 * (2008) Regularization Paths for Generalized Linear Models via Coordinate
 * Descent. http://www-stat.stanford.edu/~hastie/Papers/glmnet.pdf
 * 
 * @author Yasser Ganjisaffar
 */

public class LassoFitGenerator {
        // This module shouldn't consume more than 8GB of memory
        private static final long MAX_AVAILABLE_MEMORY = 8L * 1024 * 1024 * 1024;

        // In order to speed up the compression, we limit the number of
        // observations,
        // but this limit is dependent on the number of features that we should
        // learn
        // their weights. In other words, for learning weights of more features, we
        // need more observations.
        private static final int MAX_OBSERVATIONS_TO_FEATURES_RATIO = 10000;

        private static final double EPSILON = 1.0e-6;

        // The default number of lambda values to use
        private static final int DEFAULT_NUMBER_OF_LAMBDAS = 100;

        // Convergence threshold for coordinate descent
        // Each inner coordination loop continues until the relative change
        // in any coefficient is less than this threshold
        private static final double CONVERGENCE_THRESHOLD = 1.0e-4;

        private static final double SMALL = 1.0e-5;
        private static final int MIN_NUMBER_OF_LAMBDAS = 5;
        private static final double MAX_RSQUARED = 0.99999;

        private float[] targets;
        private float[][] observations;
        private int numFeatures;
        private int numObservations;

        public long getMaxAllowedObservations(int maxNumFeatures) {
            long maxObservations = (MAX_AVAILABLE_MEMORY / maxNumFeatures / (Float.SIZE / 8));
            if (maxObservations > MAX_OBSERVATIONS_TO_FEATURES_RATIO * maxNumFeatures) {
                    //maxObservations = MAX_OBSERVATIONS_TO_FEATURES_RATIO * maxNumFeatures;
            }
            return maxObservations;
        }

        public void init(int maxNumFeatures, int numObservations) throws Exception {
                this.numFeatures = maxNumFeatures;
                if (numObservations > getMaxAllowedObservations(maxNumFeatures)) {
                        throw new Exception("Number of observations (" + numObservations + ") exceeds the maximum allowed number: "
                                        + getMaxAllowedObservations(maxNumFeatures));
                }
                this.numObservations = numObservations;
                //observations = new float[this.numObservations][];
                observations = new float[numFeatures][];
                for (int t = 0; t < maxNumFeatures; t++) {
                        observations[t] = new float[this.numObservations];
                }
                targets = new float[this.numObservations];
        }

        public void setNumberOfFeatures(int numFeatures) {
                this.numFeatures = numFeatures;
        }

        public void setFeatureValues(int idx, float[] values) {
                for (int i = 0; i < values.length; i++) {
                        observations[idx][i] = values[i];
                }
        }
        
        public float[] getFeatureValues(int idx) {
                return observations[idx];
        }

        public void setObservationValues(int idx, float[] values) {
                for (int f = 0; f < numFeatures; f++) {
                        observations[f][idx] = values[f];
                }
        }

        private LassoFit getLassoFit(int maxAllowedFeaturesPerModel) {
                long startTime = System.currentTimeMillis();

                if (maxAllowedFeaturesPerModel < 0) {
                        maxAllowedFeaturesPerModel = numFeatures;
                }
                int numberOfLambdas = DEFAULT_NUMBER_OF_LAMBDAS;
                int maxAllowedFeaturesAlongPath = (int) Math.min(maxAllowedFeaturesPerModel * 1.2, numFeatures);

                // lambdaMin = flmin * lambdaMax
                double flmin = (numObservations < numFeatures ? 5e-2 : 1e-4);

                /********************************
                 * Standardize features and target: Center the target and features
                 * (mean 0) and normalize their vectors to have the same standard
                 * deviation
                 */
                double[] featureMeans = new double[numFeatures];
                double[] featureStds = new double[numFeatures];
                double[] feature2residualCorrelations = new double[numFeatures];

                float factor = (float) (1.0 / Math.sqrt(numObservations));
                for (int j = 0; j < numFeatures; j++) {
                        double mean = MathUtil.getAvg(observations[j]);
                        featureMeans[j] = mean;
                        for (int i = 0; i < numObservations; i++) {
                                observations[j][i] = (float) (factor * (observations[j][i] - mean));
                        }
                        featureStds[j] = Math.sqrt(MathUtil.getDotProduct(observations[j], observations[j]));

                        MathUtil.divideInPlace(observations[j], (float) featureStds[j]);
                }

                float targetMean = (float) MathUtil.getAvg(targets);
                for (int i = 0; i < numObservations; i++) {
                        targets[i] = factor * (targets[i] - targetMean);
                }
                float targetStd = (float) Math.sqrt(MathUtil.getDotProduct(targets, targets));
                MathUtil.divideInPlace(targets, targetStd);

                for (int j = 0; j < numFeatures; j++) {
                        feature2residualCorrelations[j] = MathUtil.getDotProduct(targets, observations[j]);
                }

                double[][] feature2featureCorrelations = MathUtil
                                .allocateDoubleMatrix(numFeatures, maxAllowedFeaturesAlongPath);
                double[] activeWeights = new double[numFeatures];
                int[] correlationCacheIndices = new int[numFeatures];
                double[] denseActiveSet = new double[numFeatures];

                LassoFit fit = new LassoFit(numberOfLambdas, maxAllowedFeaturesAlongPath, numFeatures);
                fit.numberOfLambdas = 0;

                double alf = Math.pow(Math.max(EPSILON, flmin), 1.0 / (numberOfLambdas - 1));
                double rsquared = 0.0;
                fit.numberOfPasses = 0;
                int numberOfInputs = 0;
                int minimumNumberOfLambdas = Math.min(MIN_NUMBER_OF_LAMBDAS, numberOfLambdas);

                double curLambda = 0;
                double maxDelta;
                for (int iteration = 1; iteration <= numberOfLambdas; iteration++) {
                        //System.out.println("Starting iteration " + iteration + " of Compression.");

                        /**********
                         * Compute lambda for this round
                         */
                        if (iteration == 1) {
                                curLambda = Double.MAX_VALUE; // first lambda is infinity
                        } else if (iteration == 2) {
                                curLambda = 0.0;
                                for (int j = 0; j < numFeatures; j++) {
                                        curLambda = Math.max(curLambda, Math.abs(feature2residualCorrelations[j]));
                                }
                                curLambda = alf * curLambda;
                        } else {
                                curLambda = curLambda * alf;
                        }

                        double prevRsq = rsquared;
                        double v;
                        while (true) {
                                fit.numberOfPasses++;
                                maxDelta = 0.0;
                                for (int k = 0; k < numFeatures; k++) {
                                        double prevWeight = activeWeights[k];
                                        double u = feature2residualCorrelations[k] + prevWeight;
                                        v = (u >= 0 ? u : -u) - curLambda;
                                        // Computes sign(u)(|u| - curLambda)+
                                        activeWeights[k] = (v > 0 ? (u >= 0 ? v : -v) : 0.0);

                                        // Is the weight of this variable changed?
                                        // If not, we go to the next one
                                        if (activeWeights[k] == prevWeight) {
                                                continue;
                                        }

                                        // If we have not computed the correlations of this
                                        // variable with other variables, we do this now and
                                        // cache the result
                                        if (correlationCacheIndices[k] == 0) {
                                                numberOfInputs++;
                                                if (numberOfInputs > maxAllowedFeaturesAlongPath) {
                                                        // we have reached the maximum
                                                        break;
                                                }
                                                for (int j = 0; j < numFeatures; j++) {
                                                        // if we have already computed correlations for
                                                        // the jth variable, we will reuse it here.
                                                        if (correlationCacheIndices[j] != 0) {
                                                                feature2featureCorrelations[j][numberOfInputs - 1] = feature2featureCorrelations[k][correlationCacheIndices[j] - 1];
                                                        } else {
                                                                // Correlation of variable with itself if one
                                                                if (j == k) {
                                                                        feature2featureCorrelations[j][numberOfInputs - 1] = 1.0;
                                                                } else {
                                                                        feature2featureCorrelations[j][numberOfInputs - 1] = MathUtil.getDotProduct(
                                                                                        observations[j], observations[k]);
                                                                }
                                                        }
                                                }
                                                correlationCacheIndices[k] = numberOfInputs;
                                                fit.indices[numberOfInputs - 1] = k;
                                        }

                                        // How much is the weight changed?
                                        double delta = activeWeights[k] - prevWeight;
                                        rsquared += delta * (2.0 * feature2residualCorrelations[k] - delta);
                                        maxDelta = Math.max((delta >= 0 ? delta : -delta), maxDelta);

                                        for (int j = 0; j < numFeatures; j++) {
                                                feature2residualCorrelations[j] -= feature2featureCorrelations[j][correlationCacheIndices[k] - 1]
                                                                * delta;
                                        }
                                }

                                if (maxDelta < CONVERGENCE_THRESHOLD || numberOfInputs > maxAllowedFeaturesAlongPath) {
                                        break;
                                }

                                for (int ii = 0; ii < numberOfInputs; ii++) {
                                        denseActiveSet[ii] = activeWeights[fit.indices[ii]];
                                }

                                do {
                                        fit.numberOfPasses++;
                                        maxDelta = 0.0;
                                        for (int l = 0; l < numberOfInputs; l++) {
                                                int k = fit.indices[l];
                                                double prevWeight = activeWeights[k];
                                                double u = feature2residualCorrelations[k] + prevWeight;
                                                v = (u >= 0 ? u : -u) - curLambda;
                                                activeWeights[k] = (v > 0 ? (u >= 0 ? v : -v) : 0.0);
                                                if (activeWeights[k] == prevWeight) {
                                                        continue;
                                                }
                                                double delta = activeWeights[k] - prevWeight;
                                                rsquared += delta * (2.0 * feature2residualCorrelations[k] - delta);
                                                maxDelta = Math.max((delta >= 0 ? delta : -delta), maxDelta);
                                                for (int j = 0; j < numberOfInputs; j++) {
                                                        feature2residualCorrelations[fit.indices[j]] -= feature2featureCorrelations[fit.indices[j]][correlationCacheIndices[k] - 1]
                                                                        * delta;
                                                }
                                        }
                                } while (maxDelta >= CONVERGENCE_THRESHOLD);

                                for (int ii = 0; ii < numberOfInputs; ii++) {
                                        denseActiveSet[ii] = activeWeights[fit.indices[ii]] - denseActiveSet[ii];
                                }
                                for (int j = 0; j < numFeatures; j++) {
                                        if (correlationCacheIndices[j] == 0) {
                                                feature2residualCorrelations[j] -= MathUtil.getDotProduct(denseActiveSet,
                                                                feature2featureCorrelations[j], numberOfInputs);
                                        }
                                }
                        }

                        if (numberOfInputs > maxAllowedFeaturesAlongPath) {
                                break;
                        }
                        if (numberOfInputs > 0) {
                                for (int ii = 0; ii < numberOfInputs; ii++) {
                                        fit.compressedWeights[iteration - 1][ii] = activeWeights[fit.indices[ii]];
                                }
                        }
                        fit.numberOfWeights[iteration - 1] = numberOfInputs;
                        fit.rsquared[iteration - 1] = rsquared;
                        fit.lambdas[iteration - 1] = curLambda;
                        fit.numberOfLambdas = iteration;

                        if (iteration < minimumNumberOfLambdas) {
                                continue;
                        }

                        int me = 0;
                        for (int j = 0; j < numberOfInputs; j++) {
                                if (fit.compressedWeights[iteration - 1][j] != 0.0) {
                                        me++;
                                }
                        }
                        if (me > maxAllowedFeaturesPerModel || ((rsquared - prevRsq) < (SMALL * rsquared))
                                        || rsquared > MAX_RSQUARED) {
                                break;
                        }
                }

                for (int k = 0; k < fit.numberOfLambdas; k++) {
                        fit.lambdas[k] = targetStd * fit.lambdas[k];
                        int nk = fit.numberOfWeights[k];
                        for (int l = 0; l < nk; l++) {
                                fit.compressedWeights[k][l] = targetStd * fit.compressedWeights[k][l] / featureStds[fit.indices[l]];
                                if (fit.compressedWeights[k][l] != 0) {
                                        fit.nonZeroWeights[k]++;
                                }
                        }
                        double product = 0;
                        for (int i = 0; i < nk; i++) {
                                product += fit.compressedWeights[k][i] * featureMeans[fit.indices[i]];
                        }
                        fit.intercepts[k] = targetMean - product;
                }

                // First lambda was infinity; fixing it
                fit.lambdas[0] = Math.exp(2 * Math.log(fit.lambdas[1]) - Math.log(fit.lambdas[2]));

                long duration = System.currentTimeMillis() - startTime;
                //System.out.println("Elapsed time for compression: " + duration);
                return fit;
        }

        public void setTargets(double[] targets) {
                for (int i = 0; i < numObservations; i++) {
                        this.targets[i] = (float) targets[i];
                }
        }

        public void setTarget(int idx, double target) {
                this.targets[idx] = (float) target;
        }

        public LassoFit fit(int maxAllowedFeaturesPerModel) {
                LassoFit fit = getLassoFit(maxAllowedFeaturesPerModel);
                /*int numberOfSolutions = fit.numberOfLambdas;
                System.out.println("Compression R2 values:");
                for (int i = 0; i < numberOfSolutions; i++) {
                        System.out.println((i + 1) + "\t" + fit.nonZeroWeights[i] + "\t"
                                        + MathUtil.getFormattedDouble(fit.rsquared[i], 4) + "\t"
                                        + MathUtil.getFormattedDouble(fit.lambdas[i], 5));
                }*/
                return fit;
        }
}