#include <stdio.h>
#include <stdlib.h>


// typedef is keyword that defines a new type
typedef struct {
    float *weights;      // Array to hold the weights
    float bias;          // Bias term
    float learning_rate; // Learning rate
    int input_size;      // Number of inputs
} Perceptron;

// Step activation function: returns 1 if input >= 0, otherwise returns 0
int activation(float sum) {
    return (sum >= 0) ? 1 : 0;
}

// Function to initialize the perceptron
Perceptron* init_perceptron(int input_size, float learning_rate) {
    Perceptron *p = (Perceptron *)malloc(sizeof(Perceptron));
    p->input_size = input_size;
    p->learning_rate = learning_rate;
    p->bias = 0.0;
    
    // Allocate memory for weights
    p->weights = (float *)malloc(input_size * sizeof(float));
    for (int i = 0; i < input_size; i++) {
        p->weights[i] = 0.0;  // Initialize weights to 0
    }
    return p;
}

// Function to calculate the weighted sum (dot product) and apply bias
float weighted_sum(Perceptron *p, int inputs[]) {
    float sum = p->bias;  // Start with bias
    for (int i = 0; i < p->input_size; i++) {
        sum += p->weights[i] * inputs[i];  // Weighted sum
    }
    return sum;
}

// Perceptron prediction: returns 1 or 0 based on the activation function
int predict(Perceptron *p, int inputs[]) {
    float sum = weighted_sum(p, inputs);
    return activation(sum);
}

// Training function
void train(Perceptron *p, int training_inputs[][2], int labels[], int num_samples, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < num_samples; i++) {
            int prediction = predict(p, training_inputs[i]);
            int error = labels[i] - prediction;
            
            // Update weights and bias
            for (int j = 0; j < p->input_size; j++) {
                p->weights[j] += p->learning_rate * error * training_inputs[i][j];
            }
            p->bias += p->learning_rate * error;  // Update bias
        }
    }
}

// Free memory allocated to perceptron
void free_perceptron(Perceptron *p) {
    free(p->weights);
    free(p);
}

int main() {
    // Training data for the AND logic gate
    int training_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    int labels[4] = {0, 0, 0, 1};  // AND output

    // Create and initialize the perceptron with 2 inputs and learning rate of 0.1
    Perceptron *p = init_perceptron(2, 0.1);

    // Train the perceptron for 10 epochs
    train(p, training_inputs, labels, 4, 10);

    // Test the perceptron
    printf("Prediction for [1, 1]: %d\n", predict(p, (int[]){1, 1}));  // Expected 1
    printf("Prediction for [0, 1]: %d\n", predict(p, (int[]){0, 1}));  // Expected 0
    printf("Prediction for [1, 0]: %d\n", predict(p, (int[]){1, 0}));  // Expected 0
    printf("Prediction for [0, 0]: %d\n", predict(p, (int[]){0, 0}));  // Expected 0

    // Free memory
    free_perceptron(p);

    return 0;
}
