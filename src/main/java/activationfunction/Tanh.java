package activationfunction;

public class Tanh implements ActivationFunction {
    /**
     * Computes the tanh activation function.
     *
     * @param x The input value
     * @return tanh(x)
     */
    @Override
    public double activate(double x) {
        return Math.tanh(x);
    }
    
    /**
     * Computes the derivative of the tanh activation function.
     *
     * @param x The input value.
     * @return The derivative of tanh(x), which is 1 - tanh(x)^2.
     */
    @Override
    public double derive(double x) {
        double tanh = Math.tanh(x);
        return 1 - tanh * tanh;
    }
}
