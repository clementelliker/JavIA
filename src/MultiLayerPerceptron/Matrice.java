package MultiLayerPerceptron;

import java.util.ArrayList;
import java.util.Random;
import java.lang.Math;


public class Matrice {
	
	public int m;
	public int n;
	public double[][] mat;
	
	public static Random seed = new Random();
	
	public Matrice(int m, int n, String type) throws Exception {
		this.m = m;
		this.n = n;
		this.mat = new double[m][];
		for(int i = 0; i < m; i++) {
			this.mat[i] = new double[n];
			for(int j = 0; j < n; j++) {
				if(type == "random") {
					this.mat[i][j] = seed.nextDouble()*2 - 1;
				}else if(type == "zeros") {
					this.mat[i][j] = 0;
				}else if(type == "ones") {
					this.mat[i][j] = 1;
				}else{
					System.out.println("Matrice type error");
					throw new Exception();
				}
				
				
			}
		}

	}
	
	public Matrice(double[][] values) {
		this.m = values.length;
		this.n = values[0].length;
		this.mat = values;
	}
	
	public void display() {
		for(int i = 0; i < this.m; i++) {
			for(int j = 0; j < this.n; j++) {
				System.out.print(this.mat[i][j] + "   ");
			}
			System.out.println();
			System.out.println();
		}
		System.out.println();
		System.out.println();
	}
	
	public Matrice dot(Matrice m2) throws Exception {
		//check dimension
		if(this.n != m2.m) {
			System.out.println("Matrice dimension not compatible");
			throw new Exception();
		}
		Matrice ret = new Matrice(this.m, m2.n, "zeros");
		for(int i = 0; i < this.m; i++) {
			for(int j = 0; j < m2.n; j++) {
				for(int k = 0; k < this.n; k++) {
					ret.mat[i][j] += this.mat[i][k]*m2.mat[k][j];
				}
			}
		}
		return ret;
	}
	
	public Matrice addOne() throws Exception {
		if(this.m != 1) {
			System.out.println("Matrice is not 1xn");
			throw new Exception();
		}
		Matrice newM = new Matrice(1,this.n+1, "zeros");
		newM.mat[0][0] = 1;
		for(int i = 1; i < newM.n; i++) {
			newM.mat[0][i] = this.mat[0][i-1];
		}
		return newM;
	}
	
	public void logisticfnc() {
		//pass the matrix through a logistic function
		for(int i = 0; i <this.m; i++) {
			for(int j = 0; j < this.n; j++) {
				this.mat[i][j] = 1/(1 + Math.pow(Math.E, -(this.mat[i][j])));
			}
		}
	}
	
	public void setToOnes() {
		for(int i = 0; i < this.m; i++) {
			for(int j = 0; j < this.n; j++) {
				this.mat[i][j] = 1;
			}
		}
	}
}
