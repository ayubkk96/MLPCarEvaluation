import java.util.Arrays;
import java.util.List;

public class NeuralNetwork {
    // 16 neurons, 21 weights each

    static final int HIDDEN_LAYER_1_SIZE = 16;
    static final int HIDDEN_LAYER_2_SIZE = 12;
    static final int OUTPUT_LAYER_SIZE = 4;
    static final double LEARNING_RATE = 0.1;

    public static void train(int epochs, List<double[]> features, List<double[]> labels) {
        double[][] weightsHidden1 = randomiseWeights(features.get(0).length, HIDDEN_LAYER_1_SIZE);
        double[] biasHidden1 = randomiseBiasHidden(HIDDEN_LAYER_1_SIZE);
        double[][] weightsHidden2 = randomiseWeights(HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE);
        double[] biasHidden2 = randomiseBiasHidden(HIDDEN_LAYER_2_SIZE);
        double[][] weightsOutputLayer = randomiseWeights(HIDDEN_LAYER_2_SIZE, OUTPUT_LAYER_SIZE);
        double[] biasOutputLayer = randomiseBiasHidden(OUTPUT_LAYER_SIZE);
        for (int epoch = 0; epoch < epochs; epoch++) {

            double totalLoss = 0;
            int correct = 0;

            for (int featureIndex = 0; featureIndex < features.size(); featureIndex++) {

                double[] feature = features.get(featureIndex);
                double[] targetOutput = labels.get(featureIndex);

                // ---- forward ----
                double[] hidden1Output = forwardPropagate(feature, weightsHidden1, biasHidden1, HIDDEN_LAYER_1_SIZE);
                double[] hidden2Output = forwardPropagate(hidden1Output, weightsHidden2, biasHidden2, HIDDEN_LAYER_2_SIZE);
                double[] output = forwardPropagate(hidden2Output, weightsOutputLayer, biasOutputLayer, OUTPUT_LAYER_SIZE);

                // ---- loss ----
                totalLoss += meanSquaredError(targetOutput, output);

                // ---- accuracy ----
                if (argMax(output) == argMax(targetOutput)) {
                    correct++;
                }

                // ---- deltas ----
                double[] deltaOutput = getDeltaOutputLayer(targetOutput, output);
                double[] deltaHidden2 = getDeltaHiddenLayer(deltaOutput, weightsOutputLayer, hidden2Output);
                double[] deltaHidden1 = getDeltaHiddenLayer(deltaHidden2, weightsHidden2, hidden1Output);

                // ---- updates ----
                updateLayerWeightsAndBiases(LEARNING_RATE, deltaOutput, hidden2Output, weightsOutputLayer, biasOutputLayer);
                updateLayerWeightsAndBiases(LEARNING_RATE, deltaHidden2, hidden1Output, weightsHidden2, biasHidden2);
                updateLayerWeightsAndBiases(LEARNING_RATE, deltaHidden1, feature, weightsHidden1, biasHidden1);
            }

            double avgLoss = totalLoss / features.size();
            double accuracy = (double) correct / features.size();

            System.out.printf(
                    "Epoch %d | Loss: %.6f | Accuracy: %.2f%%%n",
                    epoch + 1,
                    avgLoss,
                    accuracy * 100
            );
        }
    }

    private static void updateLayerWeightsAndBiases(
            double learningRate,
            double[] deltaCurrentLayer,
            double[] prevLayerOutput,
            double[][] weightsCurrentLayer,
            double[] biasCurrentLayer
    ) {
        for (int neuron = 0; neuron < deltaCurrentLayer.length; neuron++) {
            for (int prev = 0; prev < prevLayerOutput.length; prev++) {
                weightsCurrentLayer[neuron][prev] += learningRate
                        * deltaCurrentLayer[neuron]
                        * prevLayerOutput[prev];
            }
            biasCurrentLayer[neuron] += learningRate * deltaCurrentLayer[neuron];
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

    private static double[] getDeltaHiddenLayer(double[] deltaNextLayer, double[][] weightsNextLayer,
                                                double[] currentLayerOutput) {
        double[] deltaCurrentLayer = new double[currentLayerOutput.length];
        for (int i = 0; i < currentLayerOutput.length; i++) {
            double sum = 0;
            for (int o = 0; o < deltaNextLayer.length; o++) {
                sum += weightsNextLayer[o][i] * deltaNextLayer[o];
            }
            deltaCurrentLayer[i] = sigmoidDerivative(currentLayerOutput[i]) * sum;
        }
        return deltaCurrentLayer;
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

    private static double meanSquaredError(double[] target, double[] output) {
        double sum = 0;
        for (int i = 0; i < target.length; i++) {
            double diff = target[i] - output[i];
            sum += diff * diff;
        }
        return sum / target.length;
    }

    private static int argMax(double[] values) {
        int index = 0;
        double max = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > max) {
                max = values[i];
                index = i;
            }
        }
        return index;
    }
}