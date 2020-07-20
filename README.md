# Machine Learning

Machine Learning projects done in University.

## Handwritten digits

Two databases of handwritten digits are used, one for training and another one for testing. Each digit is represented as a 8x8 matrix, with each cell having 16 possible values. Each digit consists, therefore, of 64 features.

![1](https://raw.githubusercontent.com/FedericoGarciaGarcia/MachineLearning/master/files/informe/Pictures/digitos.png)

To improve performance, dimensionality will be reduced.

### 3D Plot

The 64 features have been reduced to 3: vertical symmetry, upper half intensity and lower half intensity.

Each axis is a different feature. Each digit is plotted with a different color.

![2](https://raw.githubusercontent.com/FedericoGarciaGarcia/MachineLearning/master/files/informe/Pictures/nube3d.png)

The *k-NN Leave One Out* algorithm was run on the training database (instead of the test database to avoid bias) to determine the quality of the reduction.

The accuracy rate was 42.18%. 

### Sum of intensities

The 64 features have been reduced to 16: sum of intensities for each row and column.

Rows are shown in blue, and columns are shown in red.

![3](https://raw.githubusercontent.com/FedericoGarciaGarcia/MachineLearning/master/files/informe/Pictures/histogramas.png)

The *k-NN Leave One Out* algorithm was run, giving an accuracy rate of 95.23%. 

### Training

Logistic Regression was used with the sum of intensities dimensionality reduction for 200 iterations.

The results were:

*E<sub>out</sub>* = 0.09961 (accuracy rate is 90.039%)

*E<sub>in</sub>* = 0.08553 (accuracy rate is 91.447%)