#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 

// #define acts as a text subsitution tool.
// Whenever the macro name is seen the preprocessor replaces 
// that name  with the corresponding value
#define INPUT_NEURONS 3
#define HIDDEN_NEURONS 3
#define OUTPUT_NEURONS 1
#define LEARNING_RATE 0.1


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}


typedef struct {
    double weights[INPUT_NEURONS]; 
    double bias;
} Neuron;


typedef struct {
    Neuron neurons[HIDDEN_NEURONS]; 
} Layer;


// RAND_MAX is a macro is the C standard library
// It is usually set to the maximum value of signed 32-it interger 
// which is RAND_MAX: 2147483647

//When using the rand() function it is automatically seeded at a default seed
//most likely 0

// the operation * 2 - 1 is performed to but the random value in the range -1 to 1

// *neuron means you are passing a pointer to the Neuron strucutre not a copy of it.
// Because the function receives the address of the Neuron, it can directly modify the original Neuron in memory.

void initialize_neuron(Neuron *neuron) {
    for (int i = 0; i < INPUT_NEURONS; i++) {
        neuron->weights[i] = ((double)rand() / RAND_MAX) * 2 - 1; 
    }
    neuron->bias = ((double)rand() / RAND_MAX) * 2 - 1;
}

void initialize_layer(Layer *layer) {
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        initialize_neuron(&(layer->neurons[i]));
    }
}

//neuron->bias is equivalent to (*neuron).bias
//The function returns the output of the sigmoid function for a single neuron
// by adding the bias and the sum of input*weights
double feedforward(Neuron *neuron, double inputs[]) {
    double activation = neuron->bias;
    for (int i = 0; i < INPUT_NEURONS; i++) {
        activation += neuron->weights[i] * inputs[i];
    }
    return sigmoid(activation);
}

// applying the feedforward function to every neuron in the layer
// &(layer->neurons[i]) is getting the address of the specific neuron in the layer
void forward_propagation(Layer *layer, double inputs[], double outputs[]) {
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        outputs[i] = feedforward(&(layer->neurons[i]), inputs);
    }
}


void backward_propagation(Layer *layer, double inputs[], double target[], double outputs[], double error_gradient[]) {
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        double error = target[i] - outputs[i];
        error_gradient[i] = error * sigmoid_derivative(outputs[i]);


        for (int j = 0; j < INPUT_NEURONS; j++) {
            layer->neurons[i].weights[j] += LEARNING_RATE * error_gradient[i] * inputs[j];
        }
        layer->neurons[i].bias += LEARNING_RATE * error_gradient[i];
    }
}


void test(Layer *layer, double inputs[]) {
    double outputs[HIDDEN_NEURONS];
    forward_propagation(layer, inputs, outputs);

    printf("Predicted output: ");
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        printf("%lf ", outputs[i]);
    }
    printf("\n");
}

