/**
 * Implements the cross-entropy loss function for neural networks.
 */
public class CrossEntropyLoss implements LossFunction {
    /**
     * Calculates the cross-entropy loss for a single set of predictions and targets.
     * It measures the performance of a classification model whose output is a probability value between 0 and 1.
     *
     * @param predicted An array of predicted probabilities, one for each class.
     * @param target An array of actual target probabilities, typically one-hot encoded.
     * @return The cross-entropy loss for the predictions.
     */
    @Override
    public double calculateLoss(double[] predicted, double[] target) {
        double loss = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            loss -= target[i] * Math.log(predicted[i]);
        }
        return loss;
    }
    
    /**
     * Calculates the total cross-entropy loss for a batch of predictions and targets.
     *
     * @param predicted A 2D array where each inner array contains predicted probabilities for a batch of data.
     * @param target A 2D array where each inner array contains the actual probabilities for the batch of data, typically one-hot encoded.
     * @return The total cross-entropy loss for the batch.
     */
    @Override
    public double calculateLoss(double[][] predicted, double[][] target) {
        double totalLoss = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double loss = 0.0;
            for (int j = 0; j < predicted[i].length; j++) {
                loss -= target[i][j] * Math.log(predicted[i][j]);
            }
            totalLoss += loss;
        }
        return totalLoss;
    }
    
    /**
     * Derives the gradient of the loss function for a single prediction compared to the target.
     *
     * @param predicted An array of predicted probabilities.
     * @param target An array of actual probabilities, typically one-hot encoded.
     * @return An array containing the gradients of the loss function with respect to the predicted probabilities.
     */
    @Override
    public double[] derive(double[] predicted, double[] target) {
        double[] derivative = new double[predicted.length];
        for (int i = 0; i < predicted.length; i++) {
            derivative[i] = predicted[i] - target[i];
        }
        return derivative;
    }
    
    /**
     * Derives the gradients of the loss function for a batch of predictions compared to a batch of targets.
     *
     * @param predicted A 2D array containing predicted probabilities for each data point in the batch.
     * @param target A 2D array containing the actual probabilities for each data point in the batch, typically one-hot encoded.
     * @return A 2D array where each inner array contains the gradients of the loss with respect to the predictions of a single data point.
     */
    @Override
    public double[][] derive(double[][] predicted, double[][] target) {
        double[][] derivatives = new double[predicted.length][predicted[0].length];
        for (int i = 0; i < predicted.length; i++) {
            for (int j = 0; j < predicted[i].length; j++) {
                derivatives[i][j] = predicted[i][j] - target[i][j];
            }
        }
        return derivatives;
    }
    
}
