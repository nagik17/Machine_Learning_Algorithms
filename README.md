## **KNN:**
1.Constructed Random search CV function. Firstly, 10 random uniform hyper parameters were obtained. <br />
2.Secondly, it divides the train dataset into 3 folds using a split_data function. And the dataset is grouped accordingly to find accuracy scores. <br />
3. Same is repeated with cross validation data and accuracy scores were obtained. The mean of both train and test(cv) are returned by the function. <br />
4. To plot hyperparamter versus accuracy, to get the smooth curve, the parameters need to be sorted. so they have been converted to dictionaries and
then plotted. We obtain two curves as seen. <br />
5. The optimal k was found out using CV data, and found out to be k=44. <br />
6. Decision boundary is plotted using k=3. And the accuracy with respect to test data was obtaoned to be 96%. <br />
## **Metrics in ML:**
Custom Implementation of Accuracy, AUC, Log-Loss, Confusion_Matrix, Precision, Recall and F1-SCORE. <br />
## **Naive Baeyes:**
Here I performed Multinomial Naive Bayes on preprocessed_arrays, first converted all non-numerical features into vectors
using techniques like BoW and tfidf for essay features and One Hot encoding for categorical features. And for numerical features, I 
performed normalization. Stacking all the features and perform Multinomial NB on non numerical(Gaussian NB on numerical features)
using simple cross validation method. 
## **Logistic Regression:** 
Analysis and Results obtained: 
1. The plot above shows that log loss decreases with increase in epoch number. <br />
2. Initial Loss on train data: 9.992007221626415e-16 Initial Loss on test data: 9.992007221626415e-16 <br />
3. Accuracy score of train was obtained to be 0.95224 or 95.224% and that of test obtained was 0.95 or 95% <br />
4. The 'dst' variable shows us that the distance between the weight vector obtained from task calculations and sklearn
implementation is very less for each feature.
## ** SVM and Calibration Plots:**
Findings: <br />
the decision boundary overfits when regularization strength is high. Underfits when is too low for both the models,
namely Logistic Regression and Support Vector machines. <br />
Task 1 was about evaluating the values of calculated decision function with sklearn's in-built function
decision_function(). <br />
Task 2 was about implementing Platt's calibration technique, by changing the y values and obtaining
probabilities. Also, learnt that there are many ways to calculate probabilities. <br />
The Platt's calibration technique is a method for extracting probabilities P(Y=1/X) from SVM outputs which is
used for classification post processing. (this was understood from a research paper called 'Probabilistic
Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods' by John C.Platt)
## **Decision Trees:**
![Capture](https://github.com/nagik17/Machine_Learning_Algorithms/blob/main/3.JPG)

## **Random Forest:**
#### TASK 1: <br />
The Boston Dataset is used here which has prices of houses from various locations. It contains 506 rows and 13
features(crime-rate,age,etc)
Here I apply Bootstrapping method and then use DecisionTreeRegressor as our model. 
By using the function "generating_samples" we obtain a sample dataset from the actual given data.
The function returns a sample data of size 506 with number of columns ranging from 3 to 13. 
We create 30 samples ensuring each sample has different set of columns. <br />
By using DecisionTreeRegressor() models fit on input data column sampled on each of the 30 samples, <br />
I found MSE = 0.03526679841897236
and Out of Bag(oob) Score = 16.288305335968378
#### TASK 2: <br />
Here, I generate 35 samples and obtain 35 MSE and OOB values in the form of a list.
Next I calculate the Confidence Intervals of MSE & OOB Score taking above two lists as samples.
By refering the Central_limit_theorem notebook,
since we have no population std_dev statistic, we apply the case 2 of the conclusion given in above mentioned notebook. <br />
I have obtained: <br />
The Confidence Interval of MSE:
[ 0.0899347151986333 , 0.17783311623026857 ]
The Confidence Interval of OOB:
[ 14.32721837538061 , 15.775525827409282 ]
#### TASK 3: <br />
Using 30 models by using 30 samples in TASK-1, the predicted output for the given xq was obtained to be 18.9





