public class MathUtilities {
    
    /**
     * Normalizes a vector to unit length.
     *
     * @param vector The array representing the vector to be normalized.
     * @return The normalized vector as an array of doubles.
     */
    public static double[] normalize(double[] vector) {
        double sum = 0;
        for (double v : vector) {
            sum += v * v;
        }
        double magnitude = Math.sqrt(sum);
        for (int i = 0; i < vector.length; i++) {
            vector[i] /= magnitude;
        }
        return vector;
    }
    
    /**
     * Multiplies two matrices.
     *
     * @param matrixA The first matrix as a 2D double array.
     * @param matrixB The second matrix as a 2D double array.
     * @return The resulting matrix product as a 2D double array.
     * @throws IllegalArgumentException if the dimensions of the matrices are not compatible for multiplication.
     */
    public static double[][] matrixMultiply(double[][] matrixA, double[][] matrixB) {
        if (matrixA[0].length != matrixB.length) {
            throw new IllegalArgumentException("Incompatible matrix dimensions.");
        }
        double[][] result = new double[matrixA.length][matrixB[0].length];
        for (int i = 0; i < matrixA.length; i++) {
            for (int j = 0; j < matrixB[0].length; j++) {
                for (int k = 0; k < matrixB.length; k++) {
                    result[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }
        }
        return result;
    }
    
    /**
     * Transposes a matrix, swapping rows with columns.
     *
     * @param matrix The matrix to transpose as a 2D double array.
     * @return The transposed matrix as a 2D double array.
     */
    public static double[][] transpose(double[][] matrix) {
        double[][] transposed = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }
    
    /**
     * Adds two vectors.
     *
     * @param vectorA The first vector as an array of doubles.
     * @param vectorB The second vector as an array of doubles.
     * @return The resulting vector from the addition.
     * @throws IllegalArgumentException if the vectors have differing dimensions.
     */
    public static double[] vectorAdd(double[] vectorA, double[] vectorB) {
        if (vectorA.length != vectorB.length) {
            throw new IllegalArgumentException("Adding vectors of differing dimensions.");
        }
        int dimensions = vectorA.length;
        double[] result = new double[dimensions];
        for (int i = 0; i < vectorA.length; i++) {
            result[i] = vectorA[i] + vectorB[i];
        }
        return result;
    }
    
    /**
     * Subtracts two vectors.
     *
     * @param vectorA The first vector as an array of doubles.
     * @param vectorB The second vector as an array of doubles.
     * @return The resulting vector from the addition.
     * @throws IllegalArgumentException if the vectors have differing dimensions.
     */
    public static double[] vectorSubtract(double[] vectorA, double[] vectorB) {
        if (vectorA.length != vectorB.length) {
            throw new IllegalArgumentException("Subtracting vectors of differing dimensions.");
        }
        int dimensions = vectorA.length;
        double[] result = new double[dimensions];
        for (int i = 0; i < dimensions; i++) {
            result[i] = vectorA[i] - vectorB[i];
        }
        return result;
    }
    
    
    /**
     * Calculates the dot product of two vectors.
     *
     * @param vectorA First vector as an array of doubles.
     * @param vectorB Second vector as an array of doubles.
     * @return The scalar dot product of the two vectors.
     * @throws IllegalArgumentException if the vectors have differing dimensions.
     */
    public static double dotProduct(double[] vectorA, double[] vectorB) {
        if (vectorA.length != vectorB.length) {
            throw new IllegalArgumentException("Dot product of vectors of differing dimensions.");
        }
        double product = 0;
        int dimensions = vectorA.length;
        for (int i = 0; i < dimensions; i++) {
            product += vectorA[i] * vectorB[i];
        }
        return product;
    }
    
    /**
     * Computes the Euclidean distance between two points in multidimensional space.
     *
     * @param pointA First point as an array of doubles.
     * @param pointB Second point as an array of doubles.
     * @return The Euclidean distance between the two points.
     * @throws IllegalArgumentException if the points have differing dimensions.
     */
    public static double distance(double[] pointA, double[] pointB) {
        if (pointA.length != pointB.length) {
            throw new IllegalArgumentException("Distance of points of differing dimensions.");
        }
        double sum = 0;
        int dimensions = pointA.length;
        for (int i = 0; i < dimensions; i++) {
            double difference = pointA[i] - pointB[i];
            sum += difference * difference;
        }
        return Math.sqrt(sum);
    }
    
    /**
     * Applies the softmax function to an array of scores, scaling them so the resultant array elements
     * lie in the range (0,1) and sum to 1, allowing the scores to act as probabilities.
     *
     * @param scores Array of scores to be transformed.
     * @return The scores transformed by the softmax function.
     */
    public static double[] softmax(double[] scores) {
        int dimensions = scores.length;
        double max = Double.NEGATIVE_INFINITY;
        for (double score : scores) {
            if (score > max) {
                max = score;
            }
        }
        double[] softmax = new double[dimensions];
        double sum = 0;
        for (int j = 0; j < dimensions; j++) {
            softmax[j] = Math.exp(scores[j] - max);
            sum += softmax[j];
        }
        for (int i = 0; i < dimensions; i++) {
            softmax[i] /= sum;
        }
        return softmax;
    }
    
    /**
     * Calculate the Mean Squared Error (MSE) between predictions and targets.
     *
     * @param expected Array of target values.
     * @param outputs  Array of predicted values.
     * @return MSE as a double.
     */
    public static double meanSquaredError(double[] expected, double[] outputs) {
        double sum = 0;
        for (int i = 0; i < outputs.length; i++) {
            double difference = outputs[i] - expected[i];
            sum += difference * difference;
        }
        return sum / outputs.length;
    }
    
    /**
     * Calculate the Cross Entropy Loss between predictions and targets.
     *
     * @param expected Array of target values.
     * @param outputs  Array of predicted values.
     * @return Cross Entropy Loss as a double.
     */
    public static double crossEntropyLoss(double[] expected, double[] outputs) {
        double loss = 0.0;
        for (int i = 0; i < outputs.length; i++) {
            loss -= expected[i] * Math.log(outputs[i] + 1e-15);
        }
        return loss;
    }
    
    /**
     * Find the index of the maximum element of the array.
     *
     * @param array Input array of doubles
     * @return The index of the maximum element of the array.
     */
    public static int argMax(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
