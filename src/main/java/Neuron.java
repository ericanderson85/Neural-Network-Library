import activationfunction.ActivationFunction;

import java.util.Random;

/**
 * Represents a neuron in a neural network, encapsulating the functionality of weight management,
 * activation calculations, and updates during backpropagation.
 */
public class Neuron {
    private double[] weights;
    private double bias;
    private double activation;
    
    /**
     * Constructs a Neuron with a specified number of inputs. Initializes weights and bias.
     *
     * @param inputSize The number of inputs this neuron receives, which determines the number of weights.
     */
    public Neuron(int inputSize) {
        this.weights = new double[inputSize];
        this.bias = 0;
        initializeWeights();
    }
    
    /**
     * Initializes the neuron's weights with random values typically near zero to start training in a neutral position.
     */
    private void initializeWeights() {
        Random randomNumberGenerator = new Random();
        for (int i = 0; i < weights.length; i++) {
            // Initialize weights to random double between -0.05 and 0.05
            weights[i] = randomNumberGenerator.nextDouble() * 0.1 - 0.05;
        }
    }
    
    /**
     * Processes inputs using the neuron's weights and bias, applying an activation function to the weighted sum.
     *
     * @param inputs Array of input values that must match the number of weights.
     * @param activationFunction The activation function to apply to the input.
     * @return The activation result from the neuron.
     */
    protected double feedForward(double[] inputs, ActivationFunction activationFunction) {
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException("Input size must match the number of weights.");
        }
        double total = MathUtilities.dotProduct(weights, inputs) + bias;
        activation = activationFunction.activate(total);
        return activation;
    }
    
    /**
     * Processes batches of inputs using the neuron's weights and bias, applying an activation function.
     *
     * @param inputs A 2D array of input values where each sub-array must match the number of weights.
     * @param activationFunction The activation function to apply.
     * @return An array of activation results for each set of inputs.
     */
    protected double[] feedForward(double[][] inputs, ActivationFunction activationFunction) {
        if (inputs[0].length != weights.length) {
            throw new IllegalArgumentException("Input size must match the number of weights.");
        }
        double[] results = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            results[i] = activationFunction.activate(MathUtilities.dotProduct(weights, inputs[i]) + bias);
        }
        return results;
    }
    
    /**
     * Updates the weights and bias of the neuron based on the error delta, learning rate, and inputs.
     *
     * @param inputs Array of input values.
     * @param learningRate Learning rate for updating the weights and bias.
     * @param delta The error term for the weight update.
     */
    protected void updateWeights(double[] inputs, double learningRate, double delta) {
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException("Input size must match the number of weights.");
        }
        for (int i = 0; i < weights.length; i++) {
            weights[i] -= learningRate * delta * inputs[i];
        }
        bias -= learningRate * delta;
    }
    
    /**
     * Updates the weights and bias of the neuron for batch inputs.
     *
     * @param inputs A 2D array of input values.
     * @param learningRate Learning rate for updating the weights and bias.
     * @param deltas An array of error terms for each input in the batch.
     */
    protected void updateWeights(double[][] inputs, double learningRate, double[] deltas) {
        double[] gradientSum = new double[weights.length];
        double deltaSum = 0;
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < weights.length; j++) {
                gradientSum[j] += deltas[i] * inputs[i][j];
            }
            deltaSum += deltas[i];
        }
        
        for (int k = 0; k < weights.length; k++) {
            weights[k] -= learningRate * gradientSum[k] / inputs.length;
        }
        bias -=learningRate * deltaSum / inputs.length;
    }
    
    /**
     * Returns the weights of this neuron.
     *
     * @return An array of weights.
     */
    public double[] getWeights() {
        return weights;
    }
    
    /**
     * Sets the weights of this neuron.
     *
     * @param weights An array of new weights to set. Must be the same length as the current weights array.
     */
    public void setWeights(double[] weights) {
        if (weights == null || weights.length != this.weights.length) {
            throw new IllegalArgumentException("Length of new weights must match the existing weights.");
        }
        this.weights = weights;
    }
    
    /**
     * Returns the bias of this neuron.
     *
     * @return The current bias.
     */
    public double getBias() {
        return bias;
    }
    
    /**
     * Sets the bias of this neuron.
     *
     * @param bias The new bias value.
     */
    public void setBias(double bias) {
        this.bias = bias;
    }
    
    /**
     * Gets the last activation value computed by this neuron.
     *
     * @return The last computed activation.
     */
    public double getActivation() {
        return activation;
    }
    
    /**
     * Sets the activation value of this neuron.
     *
     * @param activation The new activation value.
     */
    public void setActivation(double activation) {
        this.activation = activation;
    }
    
}
