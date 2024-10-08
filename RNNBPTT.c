#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 3
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1
#define TIME_STEPS 5
#define LEARNING_RATE 0.01


#define MAX_LINE_LENGTH 1024
#define MAX_ROWS 100

// Structure to hold the data from CSV
typedef struct {
    char name[50];
    int age;
    float salary;
} Employee;




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
    Employee employees[MAX_ROWS];
    int row = 0;

    // Open the CSV file for reading
    file = fopen("reduced_dataset.csv", "r");

    if (file == NULL) {
        printf("Could not open file\n");
        return 1;
    }

    // Skip the header line
    fgets(buffer, MAX_LINE_LENGTH, file);

    // Read each line from the CSV
    while (fgets(buffer, MAX_LINE_LENGTH, file)) {
        // Parse each line and store in the employees array
        sscanf(buffer, "%[^,], %d, %f", employees[row].name, &employees[row].age, &employees[row].salary);
        row++;
    }

    fclose(file);

    // Print out the employees data
    for (int i = 0; i < row; i++) {
        printf("Employee %d: Name: %s, Age: %d, Salary: %.2f\n", i + 1, employees[i].name, employees[i].age, employees[i].salary);
    }
    RNN rnn;
    initialize_rnn(&rnn);

    
    double input[TIME_STEPS][INPUT_SIZE] = {
        {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}
    };
    
    double target[TIME_STEPS] = {0, 1, 0, 1, 0};

    
    for (int epoch = 0; epoch < 10000; epoch++) {
        backprop_through_time(&rnn, input, target);
    }

    
    double new_input[INPUT_SIZE] = {1, 0, 0};
    double output = forward_step(&rnn, new_input, 1);
    printf("Predicted output: %lf\n", output);

    return 0;
}