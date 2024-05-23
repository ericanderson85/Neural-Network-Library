package lossfunction;

/**
 * Implements the mean squared error loss function for neural networks.
 */
public class MeanSquaredError implements LossFunction {
    /**
     * Calculates the mean squared error for a single set of predictions and targets.
     * It is the mean of the squares of the differences between predicted and actual values.
     *
     * @param predicted An array of predicted values from the network.
     * @param target An array of actual target values that the network is expected to predict.
     * @return The mean squared error for the predictions.
     */
    @Override
    public double calculateLoss(double[] predicted, double[] target) {
        double loss = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double difference = predicted[i] - target[i];
            loss += difference * difference;
        }
        return loss / predicted.length;
    }
    
    /**
     * Calculates the mean squared error for a batch of predictions and targets.
     * This method averages the squared differences for each prediction-target pair in the batch.
     *
     * @param predicted A 2D array where each inner array contains predicted values for a batch of data.
     * @param target A 2D array where each inner array contains the actual target values for the batch.
     * @return The mean squared error averaged over the batch.
     */
    @Override
    public double calculateLoss(double[][] predicted, double[][] target) {
        double totalLoss = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double loss = 0.0;
            for (int j = 0; j < predicted[i].length; j++) {
                double difference = predicted[i][j] - target[i][j];
                loss += difference * difference;
            }
            totalLoss += loss / predicted[i].length;
        }
        return totalLoss;
    }
    
    /**
     * Derives the gradient of the loss function for a single prediction compared to the target.
     *
     * @param predicted An array of predicted values.
     * @param target An array of actual target values.
     * @return An array containing the gradients of the loss function with respect to the predicted values.
     */
    @Override
    public double[] derive(double[] predicted, double[] target) {
        double[] derivative = new double[predicted.length];
        for (int i = 0; i < predicted.length; i++) {
            derivative[i] = 2 * (predicted[i] - target[i]);
        }
        return derivative;
    }
    
    /**
     * Derives the gradients of the loss function for a batch of predictions compared to a batch of targets.
     * This method provides gradients for each prediction-target pair in the batch.
     *
     * @param predicted A 2D array containing predicted values for each data point in the batch.
     * @param target A 2D array containing the actual target values for each data point in the batch.
     * @return A 2D array where each inner array contains the gradients of the loss with respect to the predictions of a single data point.
     */
    @Override
    public double[][] derive(double[][] predicted, double[][] target) {
        double[][] derivatives = new double[predicted.length][predicted[0].length];
        for (int i = 0; i < predicted.length; i++) {
            for (int j = 0; j < predicted[i].length; j++) {
                derivatives[i][j] = 2 * (predicted[i][j] - target[i][j]);
            }
        }
        return derivatives;
    }
}
