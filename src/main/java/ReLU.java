public class ReLU implements ActivationFunction {
    /**
     * Computes the Rectified NeuralNetwork.Linear Unit activation function, which thresholds at zero.
     *
     * @param x The input value.
     * @return The output of NeuralNetwork.ReLU, which is zero if x is less than zero, and x otherwise.
     */
    @Override
    public double activate(double x) {
        return Math.max(0, x);
    }
    
    /**
     * Computes the derivative of the Rectified NeuralNetwork.Linear Unit activation function.
     *
     * @param x The input value.
     * @return The derivative of NeuralNetwork.ReLU, which is 0 if x is less than zero, and 1 otherwise.
     */
    @Override
    public double derive(double x) {
        return x <= 0 ?
               0 :
               1;
    }
    
}
