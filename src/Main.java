import java.util.Arrays;

public class Main {


    public static void main(String[] args) {
        CarDataset.fromInputStream(DataSetLoader.load("resources/car.data"));
        CarDataset.setEncodedArray();
        System.out.println(Arrays.toString(CarDataset.allFeatureVectors.get(0)));
        System.out.println(Arrays.toString(CarDataset.allLabelVectors.get(0)));
        NeuralNetwork.train(200, CarDataset.allFeatureVectors, CarDataset.allLabelVectors);
    }
}