package MultiLayerPerceptron;

import java.util.ArrayList;
import java.util.Random;

public class Main {
	
	public static void main (String[] args) throws Exception{
		
		Random seed = new Random();
		
		//DATASET
		
		int nbData = 10000;
		double[][][] inputs = new double[nbData][][];
		double[][][] outputs = new double[nbData][][];
		
		for(int i = 0; i < inputs.length; i++) {
			double x = seed.nextDouble();
			double y = seed.nextDouble();
			inputs[i] = new double[][] {{x,y}};		
			if(Math.sqrt((0.5 - y) * (0.5 - y) + (0.5 - x) * (0.5 - x)) < 0.4) { //checking if inside 0.4 radius centered in 0.5 circle
				outputs[i] = new double[][] {{1,0}};
			}else {
				outputs[i] = new double[][] {{0,1}};
			}
		}
		
		Dataset dataset = new Dataset(inputs, outputs);
		
		int[] dim = {2,10,10,10,2};
		double lR = 0.01;
		int nbIter = 100000;
		NeuralNetwork nn = new NeuralNetwork(dim, lR);
		nn.train(nbIter, dataset);
	}
	
	
}
