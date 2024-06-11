/**
 * Represents a layer in a neural network, consisting of an array of neurons and associated activation functions.
 * This class handles both feed-forward and back-propagation processes for single inputs and batch inputs.
 */
public class Layer {
    private Neuron[] neurons;
    private double[][] lastBatchActivations;
    private final ActivationFunction activationFunction;
    
    /**
     * Constructs a Layer with a specified number of neurons, input size, and activation function.
     *
     * @param layerSize The number of neurons in the layer.
     * @param inputSize The size of the input received by each neuron.
     * @param activationFunction The activation function to be used by all neurons in the layer.
     */
    public Layer(int layerSize, int inputSize, ActivationFunction activationFunction) {
        this.neurons = new Neuron[layerSize];
        this.activationFunction = activationFunction;
        initializeLayer(inputSize);
    }
    
    /**
     * Initializes all neurons in the layer with the specified input size.
     *
     * @param inputSize The size of the input that each neuron will accept.
     */
    private void initializeLayer(int inputSize) {
        for (int i = 0; i < neurons.length; i++) {
            neurons[i] = new Neuron(inputSize);
        }
    }
    
    /**
     * Performs feed-forward operation for a single set of inputs through this layer.
     *
     * @param inputs An array of input values to be processed by the layer.
     * @return An array of output values from the layer.
     */
    protected double[] feedForward(double[] inputs) {
        double[] outputs = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].feedForward(inputs, activationFunction);
        }
        return outputs;
    }
    
    /**
     * Performs feed-forward operation for a batch of inputs through this layer.
     *
     * @param inputs A 2D array of input values to be processed by the layer in batches.
     * @return A 2D array of output values from the layer.
     */
    protected double[][] feedForward(double[][] inputs) {
        double[][] outputs = new double[inputs.length][neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            double[] neuronOutputs = neurons[i].feedForward(inputs, activationFunction);
            for (int j = 0; j < inputs.length; j++) {
                outputs[j][i] = neuronOutputs[j];
            }
        }
        lastBatchActivations = outputs;
        return outputs;
    }
    
    /**
     * Performs back-propagation for a single set of errors and inputs through this layer.
     *
     * @param errors An array of error terms from the next layer.
     * @param inputs An array of input values to the layer.
     * @param learningRate The learning rate for weight updates.
     * @return An array of error terms to propagate back to the previous layer.
     */
    protected double[] backPropagate(double[] errors, double[] inputs, double learningRate) {
        double[] previousLayerErrors = new double[inputs.length];
        
        for (int j = 0; j < neurons.length; j++) {
            double delta = errors[j] * activationFunction.derive(neurons[j].getActivation());
            
            neurons[j].updateWeights(inputs, delta, learningRate);
            
            double[] weights = neurons[j].getWeights();
            for (int i = 0; i < weights.length; i++) {
                previousLayerErrors[i] += weights[i] * delta;
            }
        }
        
        return previousLayerErrors;
    }
    
    /**
     * Performs back-propagation for a batch of errors and inputs through this layer.
     *
     * @param errors A 2D array of error terms from the next layer for each input in the batch.
     * @param inputs A 2D array of input values to the layer for each input in the batch.
     * @param learningRate The learning rate for weight updates.
     * @return A 2D array of error terms to propagate back to the previous layer.
     */
    protected double[][] backPropagate(double[][] errors, double[][] inputs, double learningRate) {
        double[][] previousLayerErrors = new double[inputs.length][inputs[0].length];
        
        for (int k = 0; k < neurons.length; k++) {
            double[] deltas = new double[errors.length];
            for (int i = 0; i < errors.length; i++) {
                deltas[i] = errors[i][k] * activationFunction.derive(neurons[k].getActivation());
            }
            
            neurons[k].updateWeights(inputs, learningRate, deltas);
            
            double[] weights = neurons[k].getWeights();
            for (int i = 0; i < inputs.length; i++) {
                for (int j = 0; j < weights.length; j++) {
                    previousLayerErrors[i][j] += weights[j] * deltas[i];
                }
            }
        }
        return previousLayerErrors;
    }
    
    /**
     * Returns the array of neurons in this layer.
     *
     * @return An array of {@link Neuron} objects representing the neurons in this layer.
     */
    public Neuron[] getNeurons() {
        return neurons;
    }
    
    /**
     * Sets the neurons in this layer.
     *
     * @param neurons An array of {@link Neuron} objects to replace the current neurons.
     * @throws IllegalArgumentException if the new array of neurons does not match the size of the existing array.
     */
    public void setNeurons(Neuron[] neurons) {
        if (neurons == null || neurons.length != this.neurons.length) {
            throw new IllegalArgumentException("New neuron array must match the size of the existing array.");
        }
        this.neurons = neurons;
    }
    
    /**
     * Retrieves the activations from the last forward pass of this layer.
     *
     * @return An array of activation values from the last forward pass.
     */
    public double[] getActivations() {
        double[] lastActivations = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            lastActivations[i] = neurons[i].getActivation();
        }
        return lastActivations;
    }
    
    /**
     * Retrieves the activations from the last batch forward pass of this layer.
     *
     * @return A 2D array of activation values from the last batch forward pass.
     */
    public double[][] getLastBatchActivations() {
        return lastBatchActivations;
    }
}
