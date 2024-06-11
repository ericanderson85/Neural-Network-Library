import java.io.*;
import java.util.Arrays;
import java.util.Random;

/**
 * Represents a feed-forward neural network with multiple layers including an output layer with a linear activation function.
 * The network is capable of training on batch and incremental data, using backpropagation to update weights.
 */
public class NeuralNetwork implements Serializable {
    private Layer[] layers;
    private final ActivationFunction activationFunction;
    private final ActivationFunction OUTPUT_ACTIVATION = new Linear();
    private final LossFunction lossFunction;
    
    /**
     * Constructs a NeuralNetwork with specified layer sizes and activation functions.
     *
     * @param inputSize The number of neurons in the input layer.
     * @param hiddenLayers An array containing the sizes of each hidden layer.
     * @param outputSize The number of neurons in the output layer.
     * @param activationFunction The activation function for all layers except the output layer.
     * @param lossFunction The loss function to use during training.
     */
    public NeuralNetwork(int inputSize, int[] hiddenLayers, int outputSize, ActivationFunction activationFunction,
                         LossFunction lossFunction) {
        this.activationFunction = activationFunction;
        this.lossFunction = lossFunction;
        createLayers(inputSize, hiddenLayers, outputSize);
    }
    
    /**
     * Initializes the layers of the network.
     *
     * @param inputSize The number of inputs for the first layer.
     * @param hiddenLayers Sizes of each hidden layer.
     * @param outputSize The number of outputs for the final layer.
     */
    private void createLayers(int inputSize, int[] hiddenLayers, int outputSize) {
        layers = new Layer[hiddenLayers.length + 1];  // + 1 for output layer
        int previousLayerSize = inputSize;
        for (int i = 0; i < hiddenLayers.length; i++) {
            layers[i] = new Layer(hiddenLayers[i], previousLayerSize, activationFunction);
            previousLayerSize = hiddenLayers[i];
        }
        
        this.layers[hiddenLayers.length] = new Layer(outputSize, previousLayerSize, OUTPUT_ACTIVATION);
    }
    
    /**
     * Trains the neural network using provided input data and expected outputs.
     *
     * @param inputs The input data for training.
     * @param expectedOutputs The expected output data for training.
     * @param epochs The number of epochs to train for.
     * @param learningRate The learning rate used for training.
     */
    public void train(double[][] inputs, double[][] expectedOutputs, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            scrambleData(inputs, expectedOutputs);
            double totalLoss = 0;
            
            for (int i = 0; i < inputs.length; i++) {
                double[] output = predict(inputs[i]);
                backPropagate(expectedOutputs[i], output, inputs[i], learningRate);
                totalLoss += lossFunction.calculateLoss(output, expectedOutputs[i]);
            }
            
            double averageLoss = totalLoss / inputs.length;
            System.out.println("Epoch " + (epoch + 1) + ": Loss = " + averageLoss);
        }
    }
    
    /**
     * Trains the neural network in batches.
     *
     * @param inputs The input data for training.
     * @param expectedOutputs The expected output data for training.
     * @param epochs The number of epochs to train for.
     * @param learningRate The learning rate used for training.
     * @param batchSize The size of each batch for training.
     */
    public void train(double[][] inputs, double[][] expectedOutputs, int epochs, double learningRate, int batchSize) {
        int numBatches = (inputs.length + batchSize - 1) / batchSize;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            scrambleData(inputs, expectedOutputs);
            double totalLoss = 0;
            
            for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
                int start = batchIndex * batchSize;
                int end = Math.min(start + batchSize, inputs.length);
                double[][] batchInputs = Arrays.copyOfRange(inputs, start, end);
                double[][] batchExpected = Arrays.copyOfRange(expectedOutputs, start, end);
                
                double[][] batchOutputs = predict(batchInputs);
                backPropagate(batchExpected, batchOutputs, batchInputs, learningRate);
                totalLoss += lossFunction.calculateLoss(batchOutputs, batchExpected);
            }
            double averageLoss = totalLoss / inputs.length;
            System.out.println((epoch + 1) + "/" + epochs + ": Loss = " + averageLoss);
        }
    }
    
    /**
     * Predicts the output for a single input.
     *
     * @param inputs The input values.
     * @return The output values as predicted by the network.
     */
    public double[] predict(double[] inputs) {
        double[] outputs = inputs;
        for (Layer layer : layers) {
            outputs = layer.feedForward(outputs);
        }
        outputs = MathUtilities.softmax(outputs);
        return outputs;
    }
    
    /**
     * Predicts the output for multiple inputs.
     *
     * @param inputs The array of input values.
     * @return The array of output values as predicted by the network.
     */
    public double[][] predict(double[][] inputs) {
        double[][] outputs = inputs;
        for (Layer layer : layers) {
            outputs = layer.feedForward(outputs);
        }
        
        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = MathUtilities.softmax(outputs[i]);
        }
        return outputs;
    }
    
    /**
     * Implements the backpropagation algorithm for single training example updates.
     *
     * @param expected The expected output values.
     * @param output The actual output from the forward pass.
     * @param initialInputs The input values to the network.
     * @param learningRate The learning rate used for weight updates.
     */
    private void backPropagate(double[] expected, double[] output, double[] initialInputs, double learningRate) {
        double[] errors = lossFunction.derive(output, expected);
        for (int i = layers.length - 1; i > 0; i--) {
            errors = layers[i].backPropagate(errors, layers[i - 1].getActivations(), learningRate);
        }
        layers[0].backPropagate(errors, initialInputs, learningRate);
    }
    
    /**
     * Implements the backpropagation algorithm for batch updates.
     *
     * @param expected The expected output values for each example in the batch.
     * @param output The actual output from the forward pass for each example in the batch.
     * @param initialInputs The input values to the network for each example in the batch.
     * @param learningRate The learning rate used for weight updates.
     */
    private void backPropagate(double[][] expected, double[][]output, double[][] initialInputs, double learningRate) {
        double[][] errors = lossFunction.derive(output, expected);
        for (int i = layers.length - 1; i > 0; i--) {
            errors = layers[i].backPropagate(errors, layers[i - 1].getLastBatchActivations(), learningRate);
        }
        layers[0].backPropagate(errors, initialInputs, learningRate);
    }
    
    /**
     * Scrambles the order of data and associated outputs to ensure randomized training inputs.
     *
     * @param inputs The input data to scramble.
     * @param outputs The output data to scramble along with inputs.
     */
    private static void scrambleData(double[][] inputs, double[][] outputs) {
        if (inputs.length != outputs.length)
            throw new IllegalArgumentException("Inputs and outputs must have the same length");
        
        Random rng = new Random();
        for (int i = inputs.length - 1; i > 0; i--) {
            int index = rng.nextInt(i + 1);
            
            double[] tempInput = inputs[index];
            inputs[index] = inputs[i];
            inputs[i] = tempInput;
            
            double[] tempOutput = outputs[index];
            outputs[index] = outputs[i];
            outputs[i] = tempOutput;
        }
    }
    
    /**
     * Saves the neural network to a file.
     * @param filename The name of the file to save the network.
     * @throws IOException if an I/O error occurs while writing the file.
     */
    public void save(String filename) throws IOException {
        try (ObjectOutputStream outputStream = new ObjectOutputStream(new FileOutputStream(filename))) {
            outputStream.writeObject(this);
        }
    }
    
    /**
     * Loads a neural network from a file.
     * @param filename The name of the file to load the network from.
     * @return The loaded neural network.
     * @throws IOException if an I/O error occurs while reading the file.
     * @throws ClassNotFoundException if the class of a serialized object cannot be found.
     */
    public static NeuralNetwork load(String filename) throws IOException, ClassNotFoundException {
        try (ObjectInputStream inputStream = new ObjectInputStream(new FileInputStream(filename))) {
            return (NeuralNetwork) inputStream.readObject();
        }
    }
}
