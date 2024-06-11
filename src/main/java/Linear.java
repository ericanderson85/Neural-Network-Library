public class Linear implements ActivationFunction {
    /**
     * Computes the linear activation function.
     *
     * @param x The input value.
     * @return The value of x
     */
    public double activate(double x) {
        return x;
    }
    
    /**
     * Computes the derivative of the linear activation function.
     *
     * @param x The input value.
     * @return The derivative of x, which is 1
     */
    public double derive(double x) {
        return 1;
    }
}
