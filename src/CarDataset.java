import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.stream.Stream;

final class CarDataset {
    // A List to hold all the car records
    public static final List<CarRecord> carRecords = new ArrayList<>();
    // A List to hold all input data (features)
    public static List<double[]> allFeatureVectors = new ArrayList<>();
    // A List to hold all output data (labels)
    public static List<double[]> allLabelVectors = new ArrayList<>();

    public static CarDataset fromInputStream(InputStream file) {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(file, StandardCharsets.UTF_8))) {
            String line;
            CarDataset ds = new CarDataset();
            while ((line = br.readLine()) != null) {
                if (line.isBlank()) continue;
                ds.addLine(line);
            }
            return ds;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void addLine(String line) {
        carRecords.add(CarParser.parse(line));
    }

    // to be refactored into a smarter way haha!
    public static void setEncodedArray() {
        for (int i = 0; i < carRecords.size(); i++) {
            double[] buyingVector = new double[4];
            double[] maintVector = new double[4];
            double[] doorsVector = new double[4];
            double[] personsVector = new double[3];
            double[] lugBootVector = new double[3];
            double[] safetyVector = new double[3];
            double[] labelVector = new double[4];
            if (carRecords.get(i).buying.equals("vhigh")) {
                buyingVector = new double[]{1, 0, 0, 0};
            }
            if (carRecords.get(i).buying.equals("high")) {
                buyingVector = new double[]{0, 1, 0, 0};
            }
            if (carRecords.get(i).buying.equals("med")) {
                buyingVector = new double[]{0, 0, 1, 0};
            }
            if (carRecords.get(i).buying.equals("low")) {
                buyingVector = new double[]{0, 0, 0, 1};
            }
            if (carRecords.get(i).maint.equals("vhigh")) {
                maintVector = new double[]{1, 0, 0, 0};
            }
            if (carRecords.get(i).maint.equals("high")) {
                maintVector = new double[]{0, 1, 0, 0};
            }
            if (carRecords.get(i).maint.equals("med")) {
                maintVector = new double[]{0, 0, 1, 0};
            }
            if (carRecords.get(i).maint.equals("low")) {
                maintVector = new double[]{0, 0, 0, 1};
            }
            if (carRecords.get(i).doors.equals("2")) {
                doorsVector = new double[]{1, 0, 0, 0};
            }
            if (carRecords.get(i).doors.equals("3")) {
                doorsVector = new double[]{0, 1, 0, 0};
            }
            if (carRecords.get(i).doors.equals("4")) {
                doorsVector = new double[]{0, 0, 1, 0};
            }
            if (carRecords.get(i).doors.equals("5more")) {
                doorsVector = new double[]{0, 0, 0, 1};
            }
            if (carRecords.get(i).persons.equals("2")) {
                personsVector = new double[]{1, 0, 0};
            }
            if (carRecords.get(i).persons.equals("4")) {
                personsVector = new double[]{0, 1, 0};
            }
            if (carRecords.get(i).persons.equals("more")) {
                personsVector = new double[]{0, 0, 1};
            }
            if (carRecords.get(i).lugBoot.equals("small")) {
                lugBootVector = new double[]{1, 0, 0};
            }
            if (carRecords.get(i).lugBoot.equals("med")) {
                lugBootVector = new double[]{0, 1, 0};
            }
            if (carRecords.get(i).lugBoot.equals("big")) {
                lugBootVector = new double[]{0, 0, 1};
            }
            if (carRecords.get(i).safety.equals("low")) {
                safetyVector = new double[]{1, 0, 0};
            }
            if (carRecords.get(i).safety.equals("med")) {
                safetyVector = new double[]{0, 1, 0};
            }
            if (carRecords.get(i).safety.equals("high")) {
                safetyVector = new double[]{0, 0, 1};
            }
            if (carRecords.get(i).label.equals("unacc")) {
                labelVector = new double[]{1, 0, 0, 0};
            }
            if (carRecords.get(i).label.equals("acc")) {
                labelVector = new double[]{0, 1, 0, 0};
            }
            if (carRecords.get(i).label.equals("good")) {
                labelVector = new double[]{0, 0, 1, 0};
            }
            if (carRecords.get(i).label.equals("vgood")) {
                labelVector = new double[]{0, 0, 0, 1};
            }
            double[] inputVector = Stream.of(buyingVector, maintVector, doorsVector,
                            personsVector, lugBootVector, safetyVector)
                    .flatMapToDouble(Arrays::stream)
                    .toArray();

            allFeatureVectors.add(inputVector);
            allLabelVectors.add(labelVector);
        }
    }
}
