package activationfunction;

public class Sigmoid implements ActivationFunction {
    private final int ROUNDING_THRESHOLD = 20;
    
    
    /**
     * Computes the sigmoid activation function which maps the input 'x'
     * to a value between 0 and 1, ensuring stability with bounds.
     *
     * @param x The input value
     * @return The sigmoid output ranging from 0 to 1. For x > 20, it returns 1.0; for x < -20, it returns 0.0.
     */
    @Override
    public double activate(double x) {
        if (x > ROUNDING_THRESHOLD) {
            return 1.0;
        } else if (x < -ROUNDING_THRESHOLD) {
            return 0.0;
        }
        return 1 / (1 + Math.exp(-x));
    }
    
    /**
     * Computes the derivative of the sigmoid activation function.
     *
     * @param x The input value.
     * @return The derivative of the sigmoid function.
     */
    @Override
    public double derive(double x) {
        if (Math.abs(x) > ROUNDING_THRESHOLD) {
            return 0.0;
        }
        double sigmoid = activate(x);
        return sigmoid * (1 - sigmoid);
    }
}
