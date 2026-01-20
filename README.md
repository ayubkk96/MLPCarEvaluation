Training a neural network to categorise a car's quality based on training data sets provided by UCI Machine Learning Repository https://archive.ics.uci.edu/dataset/19/car+evaluation


Here’s a clean, professional README for the MLP-only repository, focused on the machine learning implementation, not the web app.
No emojis, portfolio-ready, and clearly separated in scope.

⸻

Multilayer Perceptron (MLP) – Car Evaluation Classifier

This repository contains a from-scratch implementation of a Multilayer Perceptron (MLP) in Java, trained to classify cars based on their attributes using the UCI Car Evaluation dataset.

The project focuses on understanding and implementing neural networks without external ML libraries, covering the full pipeline from data parsing and encoding to training, evaluation, and model serialization.

⸻

Project Overview

The goal of this project is to build and train a neural network that predicts one of four car evaluation classes:
	•	unacc – Unacceptable
	•	acc – Acceptable
	•	good – Good
	•	vgood – Very Good

Predictions are made based on six categorical car attributes, encoded into numerical feature vectors.

⸻

Key Objectives
	•	Implement an MLP from first principles
	•	Understand forward propagation and backpropagation
	•	Perform feature engineering using one-hot encoding
	•	Train and evaluate a classification model
	•	Persist trained model weights for later inference
	•	Keep the ML logic framework-free and transparent

⸻

Dataset
	•	Source: UCI Machine Learning Repository – Car Evaluation Dataset
	•	Instances: 1,728
	•	Features: 6 categorical attributes
	•	Classes: 4

Input Attributes
	•	Buying price
	•	Maintenance cost
	•	Number of doors
	•	Passenger capacity
	•	Luggage boot size
	•	Safety rating

Each attribute is one-hot encoded, resulting in 21 input features.

⸻

Model Architecture
	•	Model type: Multilayer Perceptron (MLP)
	•	Input layer: 21 neurons
	•	Hidden layer 1: 16 neurons
	•	Hidden layer 2: 12 neurons
	•	Output layer: 4 neurons
	•	Activation function: Sigmoid
	•	Loss function: Mean Squared Error (MSE)
	•	Optimisation: Gradient descent via backpropagation

⸻

Training Process
	1.	Parse raw dataset rows into structured records
	2.	One-hot encode categorical features
	3.	Forward propagate inputs through the network
	4.	Compute prediction error
	5.	Backpropagate gradients
	6.	Update weights and biases
	7.	Track loss and classification accuracy per epoch

The model is trained entirely in Java without relying on external ML libraries.

⸻

Model Persistence

After training, the model’s learned parameters are saved to disk:
	•	Weight matrices
	•	Bias vectors

These files can later be loaded by an inference-only application (such as a web service) without retraining the model.

⸻

Running the Training

Prerequisites
	•	Java 21
	•	Maven

./mvnw clean package
java -jar target/*.jar

Training output includes:
	•	Epoch number
	•	Average loss
	•	Classification accuracy

⸻

Evaluation

Model performance is measured using classification accuracy on held-out test data.

Due to class imbalance in the dataset, the model tends to learn very strong decision boundaries for unacc and vgood, with good being a narrower class.

⸻

Design Decisions
	•	No ML frameworks used (e.g. TensorFlow, DL4J)
	•	Explicit matrix and vector operations
	•	Clear separation between:
	•	data parsing
	•	feature encoding
	•	model logic
	•	Emphasis on clarity and learning over performance optimisation

⸻

Relationship to Other Projects

This repository contains only the training and ML logic.

A separate repository uses the trained weights from this project to:
	•	perform inference
	•	expose predictions via a Spring Boot web application
	•	deploy the model as a cloud-based decision support system

⸻

Limitations and Future Work
	•	Replace sigmoid + MSE with softmax + cross-entropy
	•	Address dataset class imbalance
	•	Add configurable hyperparameters
	•	Improve model calibration
	•	Add confusion matrix evaluation

⸻

Why this project?

This project demonstrates:
	•	a solid understanding of neural networks
	•	the ability to implement ML algorithms from scratch
	•	comfort working at a low level with data and math
	•	the full ML lifecycle, not just model usage

⸻

License

MIT License
