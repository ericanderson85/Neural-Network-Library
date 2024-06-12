To use, clone this repository.
```
git clone https://github.com/ericanderson85/mlp/
```

Then, build the project to a jar. 
```
javac -d bin src/main/java/*
jar cvf mlp.jar  -C bin/ .
```

Import the jar into IntelliJ IDEA through project structure.


Example usage of this project with the MNIST handwriting digits dataset:
```
public class Main {
    private static final String TRAIN_IMAGES_PATH = "resources/processed_data/trainImages.ser";
    private static final String TRAIN_LABELS_PATH = "resources/processed_data/trainLabels.ser";
    private static final String TEST_IMAGES_PATH = "resources/processed_data/testImages.ser";
    private static final String TEST_LABELS_PATH = "resources/processed_data/testLabels.ser";
    
    private static final int IMAGE_WIDTH = 28;
    private static final int IMAGE_HEIGHT = 28;
    private static final int INPUT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
    private static final int OUTPUT_SIZE = 10;
    
    private static final int[] HIDDEN_LAYERS = {128, 128};
    
    private static final int TOTAL_EPOCHS = 1500;
    private static final int EPOCHS = 25;
    private static final int BATCH_SIZE = 5;
    
    private static final double INITIAL_LEARNING_RATE = 0.001;
    private static final double MINIMUM_LEARNING_RATE = 0.0005;
    private static final double LEARNING_RATE_DECAY_FACTOR = 0.8;
    
    private static double learningRate = INITIAL_LEARNING_RATE;
    
    private static final ActivationFunction ACTIVATION_FUNCTION = new Tanh();
    private static final LossFunction LOSS_FUNCTION = new CrossEntropyLoss();
    
    
    public static void main(String[] args) {
        double[][] trainImages = readData(TRAIN_IMAGES_PATH);
        double[][] trainLabels = readData(TRAIN_LABELS_PATH);
        double[][] testImages = readData(TEST_IMAGES_PATH);
        double[][] testLabels = readData(TEST_LABELS_PATH);
        
        NeuralNetwork network = new NeuralNetwork(INPUT_SIZE, HIDDEN_LAYERS, OUTPUT_SIZE, ACTIVATION_FUNCTION,
                                                  LOSS_FUNCTION);
        
        int epochCount = 0;
        while (epochCount < TOTAL_EPOCHS) {
            network.train(trainImages, trainLabels, EPOCHS, learningRate);
            
            int correct = 0;
            for (int i = 0; i < testLabels.length; i++) {
                double[] input = testImages[i];
                double[] output = network.predict(input);
                int actual = MathUtilities.argMax(testLabels[i]);
                int guess = MathUtilities.argMax(output);
                
                if (actual == guess) {
                    correct++;
                }
            }
            
            epochCount += EPOCHS;
            System.out.printf("\nEpoch %d:\n%d/%d = %.3f\nLearning rate = %.5f\n\n", epochCount, correct,
                              testLabels.length, (double) correct / testLabels.length, learningRate);
            updateLearningRate();
        }
    }
    
    private static void updateLearningRate() {
        learningRate = Math.max(learningRate * LEARNING_RATE_DECAY_FACTOR, MINIMUM_LEARNING_RATE);
    }
    
    private static double[][] readData(String fileName) {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName))) {
            return (double[][]) in.readObject();
        } catch (IOException |
                 ClassNotFoundException e) {
            System.out.println("Failed to read " + fileName + ": " + e.getMessage());
            return null;
        }
    }
}
```



