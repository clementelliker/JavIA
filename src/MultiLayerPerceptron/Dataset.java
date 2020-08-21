package MultiLayerPerceptron;

import java.util.ArrayList;
import java.util.Random;

public class Dataset {
	
	public static Random seed = new Random();
	
	public ArrayList<Data> trainingset;
	public ArrayList<Data> validSet;
	
	public Dataset(double[][][] inputs, double[][][] outputs) {
		this.trainingset = new ArrayList<Data>();
		this.validSet = new ArrayList<Data>();
		for(int i = 0; i < inputs.length; i++) {
			Data d = new Data(new Matrice(inputs[i]), new Matrice(outputs[i]));
			double rand = seed.nextDouble();
			if(rand < 0.7) this.trainingset.add(d);
			else this.validSet.add(d);	
		}
	}
}
