#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 5
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1
#define TIME_STEPS 5
#define LEARNING_RATE 0.01

#define MAX_LINE_LENGTH 1024
#define MAX_ROWS 100

typedef struct {
    char datetime[25];  // Adjusted for datetime format
    double nat_demand;
    double T2M;
    double QV2M;
    double TQL;
    double W2M;
} DataEntry;


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


double forward_step(RNN *rnn, double input[], int t) {
    double h_new[HIDDEN_SIZE] = {0};

    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = rnn->b_h[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += input[j] * rnn->W_ih[j][i];
        }
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += rnn->h[t - 1][j] * rnn->W_hh[j][i];
        }
        h_new[i] = tanh_activation(sum);
    }

    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        rnn->h[t][i] = h_new[i];
    }

    
    double output = 0;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        output += rnn->h[t][i] * rnn->W_ho[i][0];
    }
    output += rnn->b_o[0];

    return output;
}


void backprop_through_time(RNN *rnn, double input[TIME_STEPS][INPUT_SIZE], double target[TIME_STEPS]) {
    double output[TIME_STEPS] = {0};
    double error[TIME_STEPS] = {0};
    double delta_h[TIME_STEPS][HIDDEN_SIZE] = {0};

    
    for (int t = 1; t < TIME_STEPS; t++) {
        output[t] = forward_step(rnn, input[t], t);
        error[t] = target[t] - output[t];
    }

    
    for (int t = TIME_STEPS - 1; t > 0; t--) {
        
        double delta_o = error[t];

        
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            double sum = delta_o * rnn->W_ho[i][0];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                sum += delta_h[t][j] * rnn->W_hh[i][j];
            }
            delta_h[t - 1][i] = sum * tanh_derivative(rnn->h[t][i]);

            
            for (int j = 0; j < INPUT_SIZE; j++) {
                rnn->W_ih[j][i] += LEARNING_RATE * delta_h[t][i] * input[t][j];
            }
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                rnn->W_hh[j][i] += LEARNING_RATE * delta_h[t][i] * rnn->h[t - 1][j];
            }
            rnn->W_ho[i][0] += LEARNING_RATE * delta_o * rnn->h[t][i];
        }
    }
}


int main() {
    FILE *file;
    char buffer[MAX_LINE_LENGTH];
    DataEntry entries[MAX_ROWS];
    int row = 0;

    file = fopen("reduced_dataset.csv", "r");
    if (file == NULL) {
        printf("Could not open file\n");
        return 1;
    }

    fgets(buffer, MAX_LINE_LENGTH, file);

    while (fgets(buffer, MAX_LINE_LENGTH, file)) {
        if (row >= MAX_ROWS) {
            printf("Max rows exceeded\n");
            break;
        }

        if (sscanf(buffer, "%24[^,],%lf,%lf,%lf,%lf,%lf", 
                   entries[row].datetime, 
                   &entries[row].nat_demand, 
                   &entries[row].T2M, 
                   &entries[row].QV2M, 
                   &entries[row].TQL, 
                   &entries[row].W2M) != 6) {
            printf("Error parsing line %d: %s\n", row + 1, buffer);
            continue; 
        }

        row++;
    }

    fclose(file);

    
    for (int i = 0; i < 10 && i < row; i++) {
        printf("Entry %d: Datetime: %s, Nat_Demand: %.2f, T2M: %.2f, QV2M: %.6f, TQL: %.6f, W2M: %.6f\n", 
               i + 1, entries[i].datetime, entries[i].nat_demand, entries[i].T2M, entries[i].QV2M, entries[i].TQL, entries[i].W2M);
    }


    RNN rnn;
    initialize_rnn(&rnn);

    double input[TIME_STEPS][INPUT_SIZE];
    double target[TIME_STEPS];

    for (int t = 0; t < TIME_STEPS; t++) {
        input[t][0] = entries[t].nat_demand;
        input[t][1] = entries[t].T2M;
        input[t][2] = entries[t].QV2M;
        input[t][3] = entries[t].TQL;
        input[t][4] = entries[t].W2M;
        target[t] = entries[t].nat_demand;  
    }


    for (int epoch = 0; epoch < 10000; epoch++) {
        backprop_through_time(&rnn, input, target);
        if (epoch % 1000 == 0) {
            printf("Epoch %d complete\n", epoch);
        }
    }

    double new_input[INPUT_SIZE] = {entries[TIME_STEPS].nat_demand, entries[TIME_STEPS].T2M, entries[TIME_STEPS].QV2M, entries[TIME_STEPS].TQL, entries[TIME_STEPS].W2M};
    double predicted_output = forward_step(&rnn, new_input, 1);
    printf("Predicted Nat_Demand for the next time step: %lf\n", predicted_output);

    return 0;
}
