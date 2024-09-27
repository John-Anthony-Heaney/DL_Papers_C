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

void initialize_rnn(RNN *rnn) {
    for (int i = 0; i < INPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            rnn->W_ih[i][j] = ((double) rand() / RAND_MAX) * 0.2 - 0.1;

    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            rnn->W_hh[i][j] = ((double) rand() / RAND_MAX) * 0.2 - 0.1;

    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < OUTPUT_SIZE; j++)
            rnn->W_ho[i][j] = ((double) rand() / RAND_MAX) * 0.2 - 0.1;

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        rnn->b_h[i] = 0.0;
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        rnn->b_o[i] = 0.0;
    }
}