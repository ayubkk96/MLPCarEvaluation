import java.util.*;

public class Main {
    public static void main(String[] args) {
        CarDataset.fromInputStream(DataSetLoader.load("resources/car.data"));
        CarDataset.setEncodedArray();

        List<double[]> X = CarDataset.allFeatureVectors;
        List<double[]> Y = CarDataset.allLabelVectors;

        // split
        double trainRatio = 0.8;
        int n = X.size();
        int trainSize = (int) (n * trainRatio);

        List<Integer> idx = new ArrayList<>();
        for (int i = 0; i < n; i++) idx.add(i);
        Collections.shuffle(idx, new Random(42));

        List<double[]> trainX = new ArrayList<>();
        List<double[]> trainY = new ArrayList<>();
        List<double[]> testX  = new ArrayList<>();
        List<double[]> testY  = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            int k = idx.get(i);
            if (i < trainSize) {
                trainX.add(X.get(k));
                trainY.add(Y.get(k));
            } else {
                testX.add(X.get(k));
                testY.add(Y.get(k));
            }
        }

        System.out.println("Train size: " + trainX.size());
        System.out.println("Test size: " + testX.size());

        // train
        NeuralNetwork.train(200, trainX, trainY);
        NeuralNetwork.saveModel("model.txt");

        // test
        double testAcc = NeuralNetwork.evaluateAccuracy(testX, testY);
        System.out.printf("Test accuracy: %.2f%%%n", testAcc * 100);
    }
}