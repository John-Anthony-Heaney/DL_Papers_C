#include <math.h>
#include <stdlib.h>
#include <stdio.h>

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
