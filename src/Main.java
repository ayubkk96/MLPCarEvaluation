public class Main {


    public static void main(String[] args) {
        CarDataset.fromInputStream(DataSetLoader.load("resources/car.data"));
        CarDataset.setEncodedArray();
        NeuralNetwork.consumeFeaturesAndLabels(CarDataset.allFeatureVectors, CarDataset.allLabelVectors);
    }
}