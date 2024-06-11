public class LeakyReLU implements ActivationFunction {
    
    /**
     * Computes the Leaky Rectified NeuralNetwork.Linear Unit (Leaky NeuralNetwork.ReLU) activation function.
     * This function allows a small, positive gradient when the unit is not active.
     *
     * @param x The input value.
     * @return The output of Leaky NeuralNetwork.ReLU, which is 0.1x if x is less than zero, and x otherwise.
     */
    @Override
    public double activate(double x) {
        return Math.max(0.1 * x, x);
    }
    
    /**
     * Computes the derivative of the Leaky Rectified NeuralNetwork.Linear Unit activation function.
     *
     * @param x The input value.
     * @return The derivative of Leaky NeuralNetwork.ReLU, which is 0.1 if x is less than zero, and 1 otherwise.
     */
    @Override
    public double derive(double x) {
        return x < 0 ?
               0.1 :
               1;
    }
}
