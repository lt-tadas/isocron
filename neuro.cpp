	//All these libraries, and you still can't read a book
#include <iostream>		//cout, plus whatever else
#include <fstream>		//ifstream
#include <sstream>		//stringstream
#include <math.h>		//exp	
#include <random>		//mt19937, random_device
#include <algorithm>	//fill, generate
#include <iterator>		//begin, end
#include <functional>	//mersenne engine
#include <numeric>		//inner_product
#include <time.h>
#include <unistd.h>

#define alpha 0.2

using namespace std;

//Sigmoid is less mathematical and more of a general term for function behavior
//Probably any "S" shaped function would do, just that the goto is 1/(1+exp(-z))
double sigmoid(double zeta){
	return 1.0 / (1.0 + exp(-1.0*zeta));
}

//The derivation comes from multiple assumptions and mathematical properties (as seen in the MIT lecture on neural nets)
//It simplifies to a clean, short formula
double delta_sigmoid(double zeta){
	//reduces function and library calling
	double z = sigmoid(zeta);
	return z * (1 - z);
}

//Again, derived in MIT lecture videos, the error is reduced to (1/2)*(target-actual)^2
double error(int target, double actual){
	return 0.5 * ((double)target - actual) * ((double)target - actual);
}  

double delta_error(int target, double actual){
	return (double)target - actual;
}

class Layer{
	public:
	uint64_t num_nodes;
	double bias;
	vector<double> values;
	vector<double> non_sigmoided_sum;
	vector<double> sigmoided_sum;
	vector<vector<double>> weights;
	vector<vector<double>> delta_weights;
	vector<double> layer_net;
	vector<double> layer_output;

	Layer(int nodes, double bias, vector<double> values){
		//In C++, it's easier to let std libraries perform the vector filling labor
		random_device rand_gen;
		mt19937 mersenne_engine(rand_gen());
		uniform_real_distribution<double> dist(0, 1);
		auto gen = bind(dist, ref(mersenne_engine));
	
		num_nodes = nodes;
		bias = bias;
	
		int values_len = values.size(); //To minimize call-time

		//The delta weights begin with zero because we do not need to calculate the difference in weights yet. 
		delta_weights = vector<vector<double>>(num_nodes);
		for(uint32_t i = 0; i < num_nodes; ++i){
			delta_weights[i] = vector<double>(values_len);
			fill(begin(delta_weights[i]), end(delta_weights[i]), 0.0);
		}
		//The weights are completely randomized like any good, stupid neural net begins. Hence opportunity for training.
		weights = vector<vector<double>>(num_nodes);
		for(uint32_t i = 0; i < num_nodes; ++i){
			weights[i] = vector<double>(values_len);
			generate(begin(weights[i]), end(weights[i]), gen);
		}
	
		values = values;
	
		non_sigmoided_sum = vector<double>(nodes);
		sigmoided_sum = vector<double>(nodes);
		
		layer_net = vector<double>(num_nodes);
		fill(begin(layer_net), end(layer_net), 0.0);
		layer_output = vector<double>(num_nodes);
		fill(begin(layer_output), end(layer_output), 0.0);
	}	
	
	void calculate_sigmoid(){
		for(uint64_t i = 0; i < layer_net.size(); ++i){
			layer_output[i] = sigmoid(layer_net[i]);
		}
	}
	
	void calculate_sum(){
		for(uint64_t i = 0; i < num_nodes; ++i){
			layer_net[i] = inner_product(begin(values), end(values), begin(weights[i]), 0.0); 
		}	
	}

	void evaluate(){
		calculate_sum();
		calculate_sigmoid();
	}
};

void layer_two_backdrop(Layer* layer, int target){
	for(uint64_t i = 0; i < layer->weights.size(); ++i){
		for(uint64_t j = 0; j < layer->weights[i].size(); ++j){
			layer->delta_weights[i][j] = delta_sigmoid(layer->layer_output[i]) * delta_error(target, layer->layer_output[i]);
			double d_weight = layer->delta_weights[i][j] * layer->values[j];
			layer->weights[i][j] += alpha * d_weight;
		}
	}
}

void layer_one_backdrop(Layer* layer, Layer* actual){
	for(uint64_t i = 0; i < layer->weights.size(); ++i){
		for(uint64_t j = 0; j < layer->weights[i].size(); ++j){
			double d_weight = actual->delta_weights[0][i] * layer->values[j] * actual->weights[0][i] * delta_sigmoid(layer->layer_output[i]);
			layer->weights[i][j] += alpha * d_weight;
		}
	}
}


int main(){
	//HAHA THIS TOOK WAY TOO GODDAMN LONG :::::::::::))))))))))))))))))))))))))))
	ifstream file("mushroom-training.txt");
	vector<double> numbers;
	vector<double> targets;
	srand(time(NULL));
	string line;
	vector<string> line_vec = vector<string>();
	
	getline(file, line);
	line_vec.push_back(line);
	double n; 
	stringstream stream(line, ios_base::in);
	stream >> n;
	targets.push_back(n);
	if(stream.peek() == ',')
		stream.ignore();
	while(stream >> n){
		n = (n > 1) ? 1 : n;
		numbers.push_back(n);
		if(stream.peek() == ',')
			stream.ignore();
	}
	while(getline(file, line)){
		line_vec.push_back(line);
	}
	
	Layer* hidden_layer = new Layer(3, -1.0 + rand() * 2, numbers);
	Layer* output_layer = new Layer(1, -1.0 + rand() * 2, hidden_layer->layer_output); 
	

	double error_sum = 0.0;
	uint32_t run_num = 0;
	uint32_t line_num = 0;
	while(run_num < 400000){
		//Used to train with one instance at a time
		line = line_vec[line_num];
		int target;
		vector<double> training;
		double n;
		stringstream stream(line, ios_base::in);
		stream >> target;
		if(stream.peek() == ',')
				stream.ignore();
		while(stream >> n){
			n = (n > 1) ? 1 : n;
			training.push_back(n);
			if(stream.peek() == ',')
				stream.ignore();
		}
	
		hidden_layer->values = training;
		hidden_layer->evaluate();
		output_layer->values = hidden_layer->layer_output;
		output_layer->evaluate();

		layer_two_backdrop(output_layer, target);
		layer_one_backdrop(hidden_layer, output_layer);
		++line_num;
		if(line_num >= line_vec.size())
			line_num = 0;
		++run_num;
	}

	file.close();
	ifstream file_2("mushroom-testing.txt");

	vector<string> line_vec_2;
	error_sum = 0.0;
	run_num = 0;
	line_num = 0;
	while(getline(file_2, line)){
		line_vec_2.push_back(line);
	}

	cout << "SWITCHING TO TESTING DATA" << endl;

	while(1){
		//Used to train with one instance at a time
		line = line_vec_2[line_num];
		int target;
		vector<double> training;
		double n;
		stringstream stream(line, ios_base::in);
		stream >> target;
		if(stream.peek() == ',')
				stream.ignore();
		while(stream >> n){
			n = (n > 1) ? 1 : n;
			training.push_back(n);
			if(stream.peek() == ',')
				stream.ignore();
		}
	
		//After all string parsing is done, we can start calculating errors
		hidden_layer->values = training;
		hidden_layer->evaluate();
		output_layer->values = hidden_layer->layer_output;
		output_layer->evaluate();

		layer_two_backdrop(output_layer, target);
		layer_one_backdrop(hidden_layer, output_layer);
		double curr_error = error(target, output_layer->layer_output[0]);
		error_sum += curr_error;
		if(run_num % 100000 == 0){
			cout << "RUN NUMBER: " << run_num << " LOSS: " << curr_error << endl;
			cout << "TARGET v ACTUAL: " << target << " " << output_layer->layer_output[0] << endl;
			cout << "ERROR AVERAGE: " << error_sum / run_num << endl << endl;
		}
		++line_num;
		if(line_num >= line_vec_2.size())
			line_num = 0;
		++run_num;
	}
	return 0;
}
