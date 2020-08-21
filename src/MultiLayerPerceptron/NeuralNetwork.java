package MultiLayerPerceptron;
import java.util.ArrayList;


public class NeuralNetwork {
	
	public ArrayList<Matrice> layers;
	public ArrayList<Matrice> weights;
	public ArrayList<Matrice> gradients;
	public double learningRate;
	public double previousLoss;
	
	public NeuralNetwork(int[] dim, double learningRate) throws Exception {
		
		this.layers = new ArrayList<Matrice>();
		this.gradients = new ArrayList<Matrice>();
		this.weights = new ArrayList<Matrice>();
		this.learningRate = learningRate;
		this.previousLoss = 10000;
		
		for(int i = 0; i < dim.length-1; i++) {
			this.weights.add(new Matrice(dim[i] + 1, dim[i+1], "random"));
		}
		
		for(int i = 1; i < dim.length; i++) {
			this.gradients.add(new Matrice(1, dim[i], "ones"));
		}
	}
	
	public void displayNN(boolean weights, boolean layers, boolean gradients) {
		if(weights) {
			System.out.println("WEIGHTS:" + "\n");
			for(int i = 0; i < this.weights.size(); i++) {
				this.weights.get(i).display();
			}
		}
		if(layers) {
			System.out.println("LAYERS:" + "\n");
			for(int i = 0; i < this.layers.size(); i++) {
				this.layers.get(i).display();
			}
		}
		if(gradients) {
			System.out.println("GRADIENTS:" + "\n");
			for(int i = 0; i < this.gradients.size(); i++) {
				this.gradients.get(i).display();
			}
		}
	}
	
	public void getOutput(Matrice inputs) throws Exception {
		this.layers.clear();
		Matrice part = inputs.addOne(); //we add a 1 at the beginning of the layer to simulate the bias input
		this.layers.add(part);
		for(int i = 0; i < this.weights.size(); i++) {
			//this.displayNN(true, true, true);
			Matrice mat = this.layers.get(i).dot(this.weights.get(i));
			mat.logisticfnc();
			if(i != this.weights.size()-1) {
				mat = mat.addOne(); //we add a 1 at the beginning of the layer to simulate the bias input
			}
			this.layers.add(mat);
		}
	}
	
	public void updateWeights(Matrice labels) {
		this.updateGradients(labels); //we get the gradients
		//this.displayNN(true,true,false);
		for(int i = 0; i < this.weights.size(); i++) {
			for(int j = 0; j < this.weights.get(i).m; j++) {
				for(int k = 0; k < this.weights.get(i).n; k++) {
					this.weights.get(i).mat[j][k] -= this.learningRate*this.layers.get(i).mat[0][j]*this.gradients.get(i).mat[0][k]; //update the weights
				}
			}
		}
		//this.displayNN(false, false, true);
	}
	
	public void updateGradients(Matrice labels) {
		//the output(1-output) is the derivative of the logistic function 
		for(int i = this.gradients.size()-1; i >= 0; i--) {
			for(int j = 0; j < this.gradients.get(i).n; j++) {
				if(i == this.gradients.size()-1) { //gradient of the last layer depend of the label and output
					double newValue = -1*(labels.mat[0][j] - this.layers.get(i + 1).mat[0][j])*(this.layers.get(i + 1).mat[0][j]*(1 - this.layers.get(i + 1).mat[0][j])); 
					this.gradients.get(i).mat[0][j] = newValue;
				}else { //other layers depends of the next layer gradients and the weights 
					double newValue = 0;
					for(int k = 0; k < this.gradients.get(i+1).n; k++) {
						newValue += this.gradients.get(i+1).mat[0][k]*this.weights.get(i+1).mat[j+1][k];
					}
					newValue *= this.layers.get(i + 1).mat[0][j+1]*(1 - this.layers.get(i + 1).mat[0][j+1]);
					this.gradients.get(i).mat[0][j] = newValue;
				}
			}
		}
	}
	
	public int getMaxOutput() {
		int max = 0;
		for(int i = 1; i < this.layers.get(this.layers.size()-1).n; i++) {
			if(this.layers.get(this.layers.size()-1).mat[0][max] < this.layers.get(this.layers.size()-1).mat[0][i]) {
				max = i;
			}
		}
		return max;
	}
	
	public void getLoss(Dataset dataset) throws Exception {
		double loss = (double)1/(dataset.validSet.size()*dataset.validSet.get(0).targets.n);
		double sum = 0;
		int right = 0;
		for(int i = 0; i < dataset.validSet.size(); i++) {
			this.getOutput(dataset.validSet.get(i).inputs);
			int guess = this.getMaxOutput();
			if(dataset.validSet.get(i).targets.mat[0][guess] == 1) right++;
			for(int j = 0; j < dataset.validSet.get(0).targets.n; j++) {
				sum += Math.pow((dataset.validSet.get(i).targets.mat[0][j] - this.layers.get(this.layers.size()-1).mat[0][j]), 2); 
			}
		}
		this.updateLR(loss*sum, (double)100*right/dataset.validSet.size());
		this.previousLoss = loss*sum;
		
		System.out.println("MeanLoss: " + (loss*sum) + "  Percentage: " + (double)100*right/dataset.validSet.size() + " %  LearningRate: " + this.learningRate);
	}
	
	public void updateLR(double loss, double perc) {
		if(loss > this.previousLoss && perc > 75) this.learningRate = this.learningRate*0.9;
	}
	
	public void train(int iter, Dataset dataset) throws Exception {
		for(int i = 0; i < iter; i++) {
			System.out.println("Epoch n°" + i);
			for(int j = 0; j < dataset.trainingset.size(); j++) {
				//dataset.trainingset.get(j).inputs.display();
				this.getOutput(dataset.trainingset.get(j).inputs);
				this.updateWeights(dataset.trainingset.get(j).targets);
			}
			this.getLoss(dataset);
			if(this.learningRate < 0.000001) this.learningRate = 0.01; //maybe add some random to LR increment to explore other locals minimums
		}
	}
		
}
