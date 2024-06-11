/**
 * Represents an activation function in a neural network.
 * Introduces non-linearity to the model, enabling it to learn complex patterns.
 */
public interface ActivationFunction {
    
    /**
     * Applies the activation function to a single input value.
     *
     * @param x The input value to the function.
     * @return The output of the activation function.
     */
    double activate(double x);
    
    /**
     * Computes the derivative of the activation function at a given point.
     *
     * @param x The point at which the derivative is evaluated.
     * @return The derivative of the activation function.
     */
    double derive(double x);
}
