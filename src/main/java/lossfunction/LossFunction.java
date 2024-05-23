package lossfunction;

/**
 * Defines the contract for loss functions in a neural network. Loss functions measure the difference
 * between the predicted output and the target output.
 */
public interface LossFunction {
    
    /**
     * Calculates the loss for a single prediction compared to a target.
     *
     * @param predicted An array of predicted values from the network.
     * @param target An array of actual target values that the network is expected to predict.
     * @return The calculated loss as a double, representing how far off the predictions are from the target.
     */
    double calculateLoss(double[] predicted, double[] target);
    
    /**
     * Derives the gradient of the loss function for a single prediction compared to a target.
     *
     * @param predicted An array of predicted values from the network.
     * @param target An array of actual target values.
     * @return An array of derivatives, each corresponding to the gradient of the loss with respect to a predicted output.
     */
    double[] derive(double[] predicted, double[] target);
    
    /**
     * Calculates the total loss for a batch of predictions compared to a batch of targets.
     *
     * @param predicted A 2D array of predicted values, where each inner array is a set of predictions for a single input.
     * @param target A 2D array of target values, where each inner array is the set of actual target values.
     * @return The average loss for the batch as a double.
     */
    double calculateLoss(double[][] predicted, double[][] target);
    
    /**
     * Derives the gradient of the loss function for a batch of predictions compared to a batch of targets.
     *
     * @param predicted A 2D array of predicted values, each inner array a set of predictions for a single input.
     * @param target A 2D array of target values, each inner array the actual target values.
     * @return A 2D array of derivatives, where each inner array contains the gradients of the loss with respect to each predicted output in the batch.
     */
    double[][] derive(double[][] predicted, double[][] target);
}
