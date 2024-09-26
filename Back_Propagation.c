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
#define LEARNING_RATE 0.0001


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
    double weights[HIDDEN_NEURONS]; 
    double bias;
} OutputNeuron;



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

double feedforward_output(OutputNeuron *output_neuron, double hidden_outputs[]) {
    double activation = output_neuron->bias;
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        activation += output_neuron->weights[i] * hidden_outputs[i];
    }
    return sigmoid(activation);
}


// applying the feedforward function to every neuron in the layer
// &(hidden_layer->neurons[i]) is getting the address of the specific neuron in the layer
void forward_propagation(Layer *hidden_layer, OutputNeuron *output_neuron, double inputs[], double *final_output) {
    double hidden_outputs[HIDDEN_NEURONS];
    
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        hidden_outputs[i] = feedforward(&(hidden_layer->neurons[i]), inputs);
    }
    
    *final_output = feedforward_output(output_neuron, hidden_outputs);
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

void train(Layer *hidden_layer, OutputNeuron *output_neuron, double inputs[][INPUT_NEURONS], double targets[][OUTPUT_NEURONS], int num_samples) {
    double hidden_outputs[HIDDEN_NEURONS];
    double final_output;
    double error_gradient_output;

    for (int epoch = 0; epoch < 1000000; epoch++) {
        for (int sample = 0; sample < num_samples; sample++) {

            
            forward_propagation(hidden_layer, output_neuron, inputs[sample], &final_output);

            
            double error_output = targets[sample][0] - final_output;
            error_gradient_output = error_output * sigmoid_derivative(final_output);

            
            for (int i = 0; i < HIDDEN_NEURONS; i++) {
                output_neuron->weights[i] += LEARNING_RATE * error_gradient_output * hidden_outputs[i];
            }
            output_neuron->bias += LEARNING_RATE * error_gradient_output;

            
            double error_gradient_hidden[HIDDEN_NEURONS];
            for (int i = 0; i < HIDDEN_NEURONS; i++) {
                error_gradient_hidden[i] = error_gradient_output * output_neuron->weights[i] * sigmoid_derivative(hidden_outputs[i]);

                for (int j = 0; j < INPUT_NEURONS; j++) {
                    hidden_layer->neurons[i].weights[j] += LEARNING_RATE * error_gradient_hidden[i] * inputs[sample][j];
                }
                hidden_layer->neurons[i].bias += LEARNING_RATE * error_gradient_hidden[i];
            }
        }
    }
}



void test(Layer *hidden_layer, OutputNeuron *output_neuron, double inputs[]) {
    double final_output;
    forward_propagation(hidden_layer, output_neuron, inputs, &final_output);

    printf("Predicted output: %lf\n", final_output);
}



int main() {
    Layer hidden_layer;
    initialize_layer(&hidden_layer);

    OutputNeuron output_neuron;  
    initialize_neuron((Neuron *)&output_neuron); 

    double inputs[8][INPUT_NEURONS] = {
        {0, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 1, 0},
        {1, 1, 1}
    };

    double targets[8][OUTPUT_NEURONS] = {
        {0},
        {1},
        {1},
        {0},
        {1},
        {0},
        {0},
        {1}
    };

    train(&hidden_layer, &output_neuron, inputs, targets, 7);

    double new_input[INPUT_NEURONS] = {1, 0, 1};
    test(&hidden_layer, &output_neuron, new_input);

    return 0;
}


