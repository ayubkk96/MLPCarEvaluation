import java.util.List;

public class NeuralNetwork {
    // 16 neurons, 21 weights each

    static final int HIDDEN_LAYER_1_SIZE = 16;
    static final int HIDDEN_LAYER_2_SIZE = 12;
    static final int OUTPUT_LAYER_SIZE = 4;
    static final double LEARNING_RATE = 0.1;

    public static void consumeFeaturesAndLabels(List<double[]> features, List<double[]> labels) {
        double[][] weightsHidden1 = randomiseWeights(features.get(0).length, HIDDEN_LAYER_1_SIZE);
        double[] biasHidden1 = randomiseBiasHidden(HIDDEN_LAYER_1_SIZE);
        double[][] weightsHidden2 = randomiseWeights(HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE);
        double[] biasHidden2 = randomiseBiasHidden(HIDDEN_LAYER_2_SIZE);
        double[][] weightsOutputLayer = randomiseWeights(HIDDEN_LAYER_2_SIZE, OUTPUT_LAYER_SIZE);
        double[] biasOutputLayer = randomiseBiasHidden(OUTPUT_LAYER_SIZE);

        for (int featureIndex = 0; featureIndex < features.size(); featureIndex++) {
            double[] feature = features.get(featureIndex);
            //forward propagate
            double[] hidden1Output = forwardPropagate(feature, weightsHidden1,
                    biasHidden1, HIDDEN_LAYER_1_SIZE);
            double[] hidden2Output = forwardPropagate(hidden1Output, weightsHidden2,
                    biasHidden2, HIDDEN_LAYER_2_SIZE);
            double[] output = forwardPropagate(hidden2Output, weightsOutputLayer,
                    biasOutputLayer, OUTPUT_LAYER_SIZE);

            //get the target output for the current feature so we can classify
            double[] targetOutput = labels.get(featureIndex);

            //get the error and the delta for each neuron in the output layer
            double[] deltaOutput = getDeltaOutputLayer(targetOutput, output);

            //update the weights and biases for the output layer
            updateWeights(LEARNING_RATE, deltaOutput, hidden2Output, weightsOutputLayer);
            updateBias(LEARNING_RATE, deltaOutput, biasOutputLayer);

            //back propagate the error to the second hidden layer
            //tells each hidden2 neuron how much it contributed to the output error
             double[] deltaHidden2 = getDeltaHiddenLayer2(deltaOutput, weightsOutputLayer, hidden2Output);

             //back prop to hidden layer 1
            double[] deltaHidden1 = getDeltaHiddenLayer1(deltaHidden2, weightsHidden2, hidden1Output);

            //UPDATE QUICK APPARENTLY MY UPDATE WEIGHTS FUNCTION IS BACKWARDS?

        }
    }

    private static void updateBias(double learningRate, double[] delta, double[] biasOutputLayer) {
        for (int deltaIndex = 0; deltaIndex < delta.length; deltaIndex++) {
            biasOutputLayer[deltaIndex] += learningRate * delta[deltaIndex];
        }
    }

    private static void updateWeights(double learningRate, double[] delta,
                                      double[] hidden2Output, double[][] weightsOutputLayer) {
        for (int deltaIndex = 0; deltaIndex < delta.length; deltaIndex++) {
            for (int hiddenLayer2OutputNeuron = 0; hiddenLayer2OutputNeuron
                    < hidden2Output.length; hiddenLayer2OutputNeuron++) {
                weightsOutputLayer[deltaIndex][hiddenLayer2OutputNeuron]
                        += learningRate * delta[deltaIndex] * hidden2Output[hiddenLayer2OutputNeuron];
            }
        }
    }

    private static double computeDelta(double error, double output) {
        return error * sigmoidDerivative(output);
    }

    private static double sigmoidDerivative(double output) {
        return output * (1 - output);
    }

    private static double[] getDeltaOutputLayer(double[] targetOutput, double[] output) {
        double[] error = new double[output.length];
        double[] delta = new double[output.length];
        for (int i = 0; i < error.length; i++) {
            error[i] = targetOutput[i] - output[i];
            delta[i] = computeDelta(error[i], output[i]);
        }
        return delta;
    }

    // ---- delta for hidden layer 2 ----
    private static double[] getDeltaHiddenLayer2(double[] deltaOutput, double[][] weightsOutputLayer,
                                                double[] hidden2Output) {
        double[] deltaH2 = new double[HIDDEN_LAYER_2_SIZE];
        for (int h2 = 0; h2 < HIDDEN_LAYER_2_SIZE; h2++) {
            double sum = 0;
            for (int o = 0; o < OUTPUT_LAYER_SIZE; o++) {
                sum += weightsOutputLayer[o][h2] * deltaOutput[o];
            }
            deltaH2[h2] = sigmoidDerivative(hidden2Output[h2]) * sum;
        }
        return deltaH2;
    }

    // ---- delta for hidden layer 1 ----
    private static double[] getDeltaHiddenLayer1(double[] hiddenLayer2Output, double[][] weightsHiddenLayer2,
                                                double[] hidden1Output) {
        double[] deltaH1 = new double[HIDDEN_LAYER_1_SIZE];
        for (int h1 = 0; h1 < HIDDEN_LAYER_1_SIZE; h1++) {
            double sum = 0;
            for (int h2 = 0; h2 < HIDDEN_LAYER_2_SIZE; h2++) {
                sum += weightsHiddenLayer2[h2][h1] * hiddenLayer2Output[h2];
            }
            deltaH1[h1] = sigmoidDerivative(hidden1Output[h1]) * sum;
        }
        return deltaH1;
    }

    public static double[] forwardPropagate(double[] inputVector, double[][] weightsHidden,
                                            double[] biasHidden, int hiddenLayerSize) {
        double[] hiddenOutput = new double[hiddenLayerSize];
        for (int neuron = 0; neuron < hiddenOutput.length; neuron++) {
            double sum = 0;
            for (int i = 0; i < inputVector.length; i++) {
                sum += inputVector[i] * weightsHidden[neuron][i];
            }
            sum += biasHidden[neuron];
            hiddenOutput[neuron] = sigmoid(sum); // apply activation
        }
        return hiddenOutput;
    }

    static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double[][] randomiseWeights(int inputSize, int hiddenLayerSize) {
        double[][] weightsHidden1 = new double[hiddenLayerSize][inputSize];
        for (int neuronIndex = 0; neuronIndex < weightsHidden1.length; neuronIndex++) {
            for (int inputIndex = 0; inputIndex < weightsHidden1[neuronIndex].length; inputIndex++) {
                weightsHidden1[neuronIndex][inputIndex] = (-0.5 + (Math.random() * 1.0));
            }
        }
        return weightsHidden1;
    }

    public static double[] randomiseBiasHidden(int weightSize) {
        double[] biasHidden = new double[weightSize];
        for (int i = 0; i < biasHidden.length; i++) {
            biasHidden[i] = (-0.5 + (Math.random() * 1.0));
        }
        return biasHidden;
    }
}