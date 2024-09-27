#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 3
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1
#define TIME_STEPS 5
#define LEARNING_RATE 0.01


double tanh_activation(double x) {
    return tanh(x);
}


double tanh_derivative(double x) {
    return 1 - x * x;
}


typedef struct {
    double W_ih[INPUT_SIZE][HIDDEN_SIZE]; 
    double W_hh[HIDDEN_SIZE][HIDDEN_SIZE]; 
    double W_ho[HIDDEN_SIZE][OUTPUT_SIZE]; 
    double b_h[HIDDEN_SIZE]; 
    double b_o[OUTPUT_SIZE]; 
    double h[TIME_STEPS][HIDDEN_SIZE]; 
} RNN;