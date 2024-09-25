#include <stdio.h>
#include <stdlib.h>


// typedef is keyword that defines a new type
typedef struct {
    float *weights;     
    float bias;          
    float learning_rate; 
    int input_size;      
} Perceptron;


int activation(float sum) {
    return (sum >= 0) ? 1 : 0;
}

// in order to insure that the output of the relu activation function was 0 or 1 a threshold had to be added
int relu(float sum) {
    float reluFloat = (sum >= 0) ? sum : 0;
    return (reluFloat >= 0.5) ? 1 : 0;
}


Perceptron* init_perceptron(int input_size, float learning_rate) {
    Perceptron *p = (Perceptron *)malloc(sizeof(Perceptron));
    p->input_size = input_size;
    p->learning_rate = learning_rate;
    p->bias = 0.0;
    
    
    p->weights = (float *)malloc(input_size * sizeof(float));
    for (int i = 0; i < input_size; i++) {
        p->weights[i] = 0.0;  
    }
    return p;
}

float weighted_sum(Perceptron *p, int inputs[]) {
    float sum = p->bias;  
    for (int i = 0; i < p->input_size; i++) {
        sum += p->weights[i] * inputs[i]; 
    }
    return sum;
}


int predict(Perceptron *p, int inputs[]) {
    float sum = weighted_sum(p, inputs);
    //return activation(sum);
    // Switched the above activation function to relu
    return relu(sum);
}


void train(Perceptron *p, int training_inputs[][2], int labels[], int num_samples, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < num_samples; i++) {
            int prediction = predict(p, training_inputs[i]);
            int error = labels[i] - prediction;
            
            
            for (int j = 0; j < p->input_size; j++) {
                p->weights[j] += p->learning_rate * error * training_inputs[i][j];
            }
            p->bias += p->learning_rate * error;  
        }
    }
}

void free_perceptron(Perceptron *p) {
    free(p->weights);
    free(p);
}

int main() {    

    int training_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    int labels[4] = {0, 1, 0, 1};

    Perceptron *p = init_perceptron(2, 0.01);

    train(p, training_inputs, labels, 4, 100);

    printf("Prediction for [0, 0]: %d\n", predict(p, (int[]){0, 0})); 
    printf("Prediction for [0, 1]: %d\n", predict(p, (int[]){0, 1}));
    printf("Prediction for [1, 0]: %d\n", predict(p, (int[]){1, 0})); 
    printf("Prediction for [1, 1]: %d\n", predict(p, (int[]){1, 1}));

    free_perceptron(p);

    return 0;
}
