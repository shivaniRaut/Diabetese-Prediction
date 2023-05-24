# Predicting Diabetes: A Comparative Analysis of Machine Learning Algorithms
## Group 8: Chris Gast, Hussein Khalil, Priyesh Ghoradkar, Shivani Raut
### INTRODUCTION

Diabetes is a chronic disease with an increasing burden on healthcare worldwide. It is estimated that 451 million people above the age of 18 had diabetes in 2017 and that number is expected to reach 693 million by 2045, making it one of the most prevalent non-communicable diseases worldwide (1). Diabetes is characterized by high blood glucose levels, which can result in severe complications such as cardiovascular disease, blindness, renal failure, and lower limb amputations (2). Nearly a third of diabetic patients are unaware of their status due to its common asymptomatic presentation and lack of awareness of the disease (3). Early and accurate identification of diabetes is crucial for timely initiation of interventions which can help to minimize the disease progression and its associated complications. 
Diabetes is linked to a multitude of risk factors, with obesity, physical inactivity, and age being the most notable (4). Among these, obesity has been consistently demonstrated to be a major driver of insulin resistance and development of type 2 diabetes. Additionally, lifestyle factors such as physical inactivity and dietary patterns have been shown to substantially influence diabetes risk. 

In recent years, the growing availability of data has led to the emergence of machine learning (ML) algorithms as powerful tools for predicting individuals at high risk of developing diabetes. These algorithms have been widely used in medical research and have demonstrated considerable potential in identifying patterns and relationships that are not immediately discernible to human experts. For instance, a study utilizing a dataset of 17,833 responses from the National Health and Nutrition database trained and tested five different models for predicting diabetes and were able to identify diabetes with up to 82.1% accuracy (5). Another study using Pima Indians diabetes data found similar accuracy using Naïve Bayes classification for 5-factor models. In addition to these, support vector machine (SVM), decision tree(DT), logistic regression, principal component analysis (PCA), neuro fuzzy inference, quantum particle swarm optimization (QPSO) algorithm, and weighted least squares support vector machine (WLS-SVM) have all been used to predict diabetes.(6–10) These models hold promise for improving screening strategies, enhancing patient care, and informing targeted public health interventions. Various feature selection techniques have been used to select the most relevant features for diabetes diagnosis. 

Considering these findings, we propose to leverage the Behavioral Risk Factor Surveillance System (BRFSS) (11) data to test ML models for predicting diabetes. The BRFSS data, being the largest ongoing telephone-based health survey in the United States offers a wealth of information on behavioral risk factors and chronic conditions, providing a unique opportunity to examine a diverse range of features potentially associated with diabetes. Furthermore, the extensive sample size and geographic coverage of the BRFSS data allow for the development of ML models that can account variations in diabetes risk factors.

In this paper, we aim to compare the performance of multiple machine learning algorithms for predicting diabetes diagnosis using a dataset comprising relevant clinical and demographic variables. Namely, we compare logistic regression, random forests, K-nearest neighbors, and linear discriminant analysis. Our goal is to identify the most effective algorithm, with the potential to improve early detection of diabetes.

### METHODS

#### Summary of Data
For this study, we accessed a dataset of 253680 survey responses from a of the CDC’s  BRFSS 2015 data publicly available on the Kaggle website. 
The BRFSS is a prominent data source in public health research in the United States. It is the largest ongoing random-digit-dialed telephone-based health survey in the United States, collecting data from non-institutionalized adult participants (aged 18 and older) across all 50 states, the district of Columbia, and selected territories. The BRFSS primarily focuses on collecting self-reported information on health-related risk behaviors, such as dietary habits, as well as the prevalence of chronic conditions and the use of preventative healthcare services (11). 
The raw dataset includes 21 feature variables describing demographic variables, lifestyle, and health conditions and health access of respondents and 1 target variable of diabetes. 

#### Demographic:
•	Age: 13-level age category (1=18-24, 2=25-29, …, 13=80 or older)
•	Sex: 0=female, 1=male
•	Education: Education level scale 1-6
•	Income:1 = less than $10,000 5 = less than $35,000 8 = $75,000 or more
#### Lifestyle:
•	BMI: Body Mass Index (continuous)
•	Smoker: smoked at least 100 cigarettes in their lifetime, 0 = no, 1 = yes
•	PhysActivity: physical activity in the past 30 days, 0 = no, 1 = yes
•	Fruits: consumes fruit 1 or more times per day, 0 = no, 1 = yes
•	Veggies: consumes veggies 1 or more times per day, 0 = no, 1 = yes
•	HvyAlcoholConsump: adult men >= 14 drinks per week; adult women >= 7 drinks per week, 0 = no, 1 = yes
#### Health Conditions:
•	HighChol: 0 = no high cholesterol, 1 = high cholesterol
•	CholCheck: 0 = no cholesterol check in 5 years, 1 = yes cholesterol check in the last 5 years
•	HeartDiseaseorAttack: coronary heart disease (CHD) or myocardial infarction (MI), 0 = no, 1 = yes
•	GenHlth: Would you say that in general your health is: scale 1-5, 1 = excellent,
2 = very good, 3 = good, 4 = fair, 5 = poor
•	MentHlth: days of poor mental health scale 1-30, days 1 = excellent … 30 = poor
•	PhysHlth : physical illness or injury days in past 30 days, scale 1-30 1 = excellent … 30 = poor
•	DiffWalk: Do you have serious difficulty walking or climbing stairs? 0 = no, 1 = yes
•	Stroke: Have you ever had a stroke? 0 = no, 1 = yes
•	HighBP: Blood Pressure, 0 = no high BP, 1 = high BP
•	Diabetes_012: 0 = no diabetes, 1 = prediabetes, 2=diabetes
#### Healthcare Access:
•	AnyHealthcare: Have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc. 0 = no, 1 = yes
•	NodocbcCost: Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? 0 = no, 1 = yes
### Data pre-processing
In the original dataset, diabetes was represented using a three-level coding system. We re-coded the outcome variable into a binary format, where 0 denotes the absence of diabetes or prediabetes, and 1 signifies the presence of either prediabetes or diabetes. Subsequently, we converted all categorical variables into factors. Numerical variables, such as BMI, MentlHlth, and PhysHlth, were treated as continuous variables.
Following this, we ensured that the dataset contained no missing values. Lastly, we eliminated duplicate entries from the dataset, which amounted to 23,968 instances. After this preprocessing, the remaining dataset comprised 229,712 unique responses.
### Data exploration
Our first step was to perform exploratory data analysis to obtain summary statistics for each variable. We used frequencies and percentages for categorical variables and means, medians, standard deviation and range for the continuous variables. We also created visualizations such as histograms and bar charts to identify outliers and better understand the distribution of each variable and their relationships with each other. Finally, we computed correlation coefficients to understand the strength and direction of the relationships between variables and to identify potential issues and inform feature selection.

During data exploration, we noted a class imbalance for diabetes. Class imbalance of the size present in this dataset, where the minority class of the target variable is only represented in 17% of the samples, can lead to biased model predictions favoring the majority class in many model types. (12,13) Both under-sampling and over-sampling techniques were considered as solutions to this problem. Under-sampling reduces the number of majority class samples to match the minority class, but results in loss of information in the majority class that can lead to bias (12). Over-sampling prevents this loss of data, and we chose to employ random over-sampling as provided by the ROSE package in R to ensure both categories for diabetes are evenly represented. ROSE uses a smoothed bootstrap technique (13)  to generate artificial samples of both majority and minority classes, resulting in a dataset with a balanced target class. For model training and test, we used a random smaller sample of 100,000 instances from the dataset to reduce computational time. That sample was split into a 70,000-sample training set and a 30,000-sample testing set. 

After applying ROSE, we obtained a balanced training dataset with respect to the no diabetes and prediabetes/diabetes class.
Variable Selection

Important features for predicting diabetes were identified through correlations with the primary outcome, bivariate tests (chi-square for categorical and independent t-tests for continuous), and through review of the literature. The most important features were BMI, Diffwalk, Physhlth, HighChol, HighBP, Age, Education, GenHlth, and Income.
To further identify the most important features we used a Random Forest Classification model trained on 70% random sample of the smaller dataset. The model was trained with 1,000 trees and the importance of each variable was calculated. The Out-of-Bag (OOB) error rate, an unbiased estimate of the model’s prediction error, was 22.33%. Following this we generate a carriable importance plot (figure 1.) to visualize the relative importance of each predictor variable. 
Based on the results of the classification model and the extensive exploratory data analysis, we identified BMI, PhysHlth, MentHlth, Age, GenHlth, Income, HighBP, Education, DiffWalk and HighChol as predictors of interest.

### Analysis
A variety of classification models were considered to predict diabetes for each respondent. Classification models that were considered include logistic regression, linear discriminant analysis (LDA), naïve bayes, random forest, k-nearest neighbors (KNN), and support vector machines (SVM). Cross-validation was performed and the outcomes for each classification model were compared. 

### RESULTS
Logistic Regression
First, classification modeling by fitting a Logistic Regression model was employed using 9 of the 21 feature variables to predict the presence of diabetes or prediabetes. These variables included BMI, PhysHlth, MentHlth, Age, GenHlth, Income, HighBP, Education, and HighChol. After fitting the model on the training data and evaluating it on the test data, an accuracy of 72.52% was obtained, indicating a moderate level of predictive performance. 

The model demonstrated a sensitivity of 75.42%, correctly identifying most individuals with diabetes, and a specificity of 69.57%, accurately distinguishing those without the condition. The precision was found to be 71.64% while the negative predictive value was 73.53%. We also employed a k-fold cross-validation (k=10) to assess the model’s performance and reliability using the ‘caret’ package in R. The cross-validated logistic regression model achieved an average accuracy of 73.25% which is close to the results obtained.
Linear Discriminant Analysis

The application of Linear Discriminant Analysis was explored as an alternative machine learning algorithm to predict the presence of diabetes using the same set of predictor variables as the logistic regression model. The LDA model was fitted on the training data and its performance was evaluated on the test data. The LDA model yielded an accuracy of 73.71%, a sensitivity of 68.72%, specificity of 75.87%, precision of 73.65%, and a negative predictive value of 71.20%. After k-fold cross validation (k=10), using the same package employed for logistic regression, the average accuracy was 72.17%.
Support Vector Machines

Support Vector Machines (SVM) are a powerful and flexible machine learning algorithm that works well for both linearly separable and non-linearly separable data, making it a suitable choice for our problem. We used the ‘e1071’ library to train the SVM model on the training data with a radial basis function kernel. The cost and gamma parameters were set to 1 and 0.1, respectively. We then used the trained SVM model to predict diabetes for the test set. The SVM model’s accuracy was 72.74%. Its sensitivity was 67.87% percent, specificity was 77.51%, precision was 74.76% and had a 71.09% negative predictive value.  

### K-Nearest Neighbors
For K-Nearest Neighbours analysis, the first part of the code finds the optimal value of the k hyperparameter for the KNN model. The code iterates over a range of k values, computes the error rate for each k value, and plots the error rate against the k value. The optimal k value is then chosen as the value that minimizes the error rate.

In the next section KNN model with the chosen k value and evaluates its performance on the test set. The code computes the accuracy of the model as the proportion of correctly classified instances, and calculates various performance metrics such as sensitivity, precision, specificity, and F1 score. Optimization of K-value based on training set accuracy was performed, and an optimal K-value of 19 was selected for the final model.

The algorithm also performs 5-fold cross-validation to assess the generalization performance of the KNN model. Cross-validation is a technique used to estimate how well a model will perform on new data by training and testing the model on different subsets of the data. The Algorithm sets up a 5-fold cross-validation using the trainControl() function and fits the KNN model using the train() function with the k hyperparameter set to 19. The code then predicts the labels of the test set using the predict() function and evaluates the performance of the model using the confusionMatrix() function. The code computes the same performance metrics as before, including accuracy, sensitivity, precision, specificity, and F1 score. The results of the cross validation were 73.08% accuracy, 76.55% sensitivity, 71.71% precision, and 69.59% specificity.
### Random Forest
Several types of tree and forest models, including a greedy decision tree, a random forest, bagging, and a tuned random forest were investigated to both verify predictor selection for other models and to compare for prediction accuracy. A Random Forest model using all predictors performed the best, correctly predicting the presence of diabetes in the test set 77.56% of the time. This model has a slightly higher sensitivity (81.55%) than specificity (73.48%), which is desirable in a predictive model for a costly condition like diabetes, where the steps which can be taken to reduce risk have almost no cost if undertaken in a healthy person in the case of a false positive. 

This model identified BMI, PhysHlth, MentHlth, Age, GenHlth, Income, and HighBP as important predictors of the presence of diabetes. These predictors matched the important predictors utilized in our other models.

A bagging approach did not decrease the test error rate or other model evaluation metrics. A grid of parameters for the random forest model was created with different values for mtry, nodesize, and ntree, and a forest was run for each combination and the model with the lowest OOB error was selected. This tuning had negligible effects on test data performance, and the random forest with the default variables was the most successful at classifying diabetes.
### Naïve Bayes
The Naive Bayes model was used to predict diabetes in patients based on a range of features. After training the model, it achieved an accuracy of 73.39%, indicating that it correctly classified the diabetes status in approximately three-quarters of the cases. The model demonstrated a sensitivity of 74.22%, which implies that it was able to correctly identify 74.22% of the patients with diabetes. In terms of specificity, the model achieved a value of 72.54%, signifying that it correctly recognized 72.54% of the patients without diabetes.
Moreover, the model's positive predictive value (precision) stood at 73.36%, reflecting the proportion of true positive cases among the instances predicted as positive. On the other hand, the negative predictive value was 73.41%, representing the proportion of true negative cases among those predicted as negative. Lastly, the F1 score, a metric that provides a balanced view of the model's performance by considering both precision and sensitivity, was found to be 73.79%. This score highlights the overall effectiveness of the Naive Bayes model in predicting diabetes status based on the selected features.

### DISCUSSION
In this project, we aimed to compare the performance of multiple machine learning algorithms for predicting diabetes and prediabetes diagnosis using a dataset comprising relevant clinical and demographic variables. We evaluated six machine learning algorithms. 

Among the classification models explored, the random forest model demonstrated the best overall performance, with an accuracy of 77.56%, sensitivity of 81.55%, and specificity of 73.48%.  This model identified BMI, PhysHlth, MentHlth, Age, GenHlth, Income, and HighBP as important predictors of the presence of diabetes, which were consistent with the predictor variables utilized in the other models. The superior performance of the random forest model in this study is consistent with previous research that has demonstrated the effectiveness of ensemble methods in handling complex, high-dimensional data. (5,6)

Despite the random forest model's relatively strong performance, other classification models, such as logistic regression, LDA, KNN, naïve bayes, and SVM, also demonstrated moderate predictive performance with accuracies ranging from 72% to 74%. This highlights the potential for a wide range of machine learning algorithms to be employed in public health research and may encourage further exploration of these techniques for predicting chronic health conditions.

It is important to note that we used the BRFSS dataset, which has both advantages and disadvantages. The advantages include cost-effective data collection through telephone interviews, large sample sizes, comprehensive geographic coverage across the United States, and a broad range of topics covered in the survey. These factors contribute to the dataset’s utility for public health research and surveillance. However, this dataset is limited due to its self-reported nature, which may be subject to recall and social desirability bias. Additionally, response rates and coverage bias may be an issue, as the survey excludes people without phones or those who are unwilling to participate. 

The cross-sectional nature of this data is also a weakness that limits the ability to establish time-dependent relationships between diabetes/prediabetes and the predictor variables. Another limitation is the class imbalance in the original dataset, which we addressed by using the ROSE technique. While this technique has been shown to be effective in balancing classes, it may introduce artificial patterns that could affect the generalizability of the models.

### CONCLUSION
In conclusion, this analysis demonstrates the potential of machine learning algorithms, particularly the random forest model, to predict diabetes and prediabetes using demographic, lifestyle, and health conditions obtained from survey data. Future research should focus on improving the quality of input data, use longitudinal data to better establish temporal relationships between risk factors and diabetes development, and explore additional feature engineering and selection techniques. Furthermore, future studies should explore the integration of these models into clinical decision support systems to improve early detection and management of diabetes. The development of accurate predictive models for diabetes and prediabetes can contribute to better public health decision making, early intervention strategies, and targeted prevention programs aimed at reducing the burden of these chronic conditions.
 
### References
1.	Cho NH, Shaw JE, Karuranga S, Huang Y, da Rocha Fernandes JD, Ohlrogge AW, et al. IDF Diabetes Atlas: Global estimates of diabetes prevalence for 2017 and projections for 2045. Diabetes Research and Clinical Practice. 2018 Apr 1;138:271–81. 
2.	Banday MZ, Sameer AS, Nissar S. Pathophysiology of diabetes: An overview. Avicenna J Med. 2020 Oct 13;10(4):174–88. 
3.	Choi SB, Kim WJ, Yoo TK, Park JS, Chung JW, Lee Y ho, et al. Screening for Prediabetes Using Machine Learning Models. Comput Math Methods Med. 2014;2014:618976. 
4.	Wu Y, Ding Y, Tanaka Y, Zhang W. Risk Factors Contributing to Type 2 Diabetes and Recent Advances in the Treatment and Prevention. Int J Med Sci. 2014 Sep 6;11(11):1185–200. 
5.	Qin Y, Wu J, Xiao W, Wang K, Huang A, Liu B, et al. Machine Learning Models for Data-Driven Prediction of Diabetes by Lifestyle Type. Int J Environ Res Public Health. 2022 Nov 15;19(22):15027. 
6.	Chang V, Bailey J, Xu QA, Sun Z. Pima Indians diabetes mellitus classification based on machine learning (ML) algorithms. Neural Comput Appl. 2022 Mar 24;1–17. 
7.	Yue C, Xin L, Kewen X, Chang S. An Intelligent Diagnosis to Type 2 Diabetes Based on QPSO Algorithm and WLS-SVM. In: 2008 International Symposium on Intelligent Information Technology Application Workshops. 2008. p. 117–21. 
8.	Çalişir D, Doğantekin E. An automatic diabetes diagnosis system based on LDA-Wavelet Support Vector Machine Classifier. Expert Systems with Applications. 2011 Jul 1;38(7):8311–5. 
9.	Kavakiotis I, Tsave O, Salifoglou A, Maglaveras N, Vlahavas I, Chouvarda I. Machine Learning and Data Mining Methods in Diabetes Research. Computational and Structural Biotechnology Journal. 2017 Jan 1;15:104–16. 
10.	Şahan S, Polat K, Kodaz H, Güneş S. The Medical Applications of Attribute Weighted Artificial Immune System (AWAIS): Diagnosis of Heart and Diabetes Diseases. In: Jacob C, Pilat ML, Bentley PJ, Timmis JI, editors. Artificial Immune Systems. Berlin, Heidelberg: Springer; 2005. p. 456–68. (Lecture Notes in Computer Science). 
11.	Rolle-Lake L, Robbins E. Behavioral Risk Factor Surveillance System. In: StatPearls [Internet]. Treasure Island (FL): StatPearls Publishing; 2023 [cited 2023 Apr 18]. Available from: http://www.ncbi.nlm.nih.gov/books/NBK553031/
12.	Rahman MM, Davis DN. Addressing the Class Imbalance Problem in Medical Datasets. IJMLC. 2013;224–8. 
13.	Lunardon N, Menardi G, Torelli N. ROSE: a Package for Binary Imbalanced Learning. The R Journal. 2014;6(1):79. 


### TABLES AND FIGURES
#### Variable importance plot
 ![Figure 1: Variable importance plot](https://github.com/shivaniRaut/Diabetese-Prediction/assets/30024267/65bf56c0-51f8-4fea-a507-35ef90fb410b)

#### Error rate vs K value
 <img width="368" alt="Figure 2: Error rate vs K value" src="https://github.com/shivaniRaut/Diabetese-Prediction/assets/30024267/275bb9c3-9e9b-4d65-82b0-e110dd7c86cf">

#### Accuracy, Sensitivity, and Specificity of Employed models
![Figure 2 Table 1.  Accuracy, Sensitivity, and Specificity of Employed models](https://github.com/shivaniRaut/Diabetese-Prediction/assets/30024267/3c07fc3f-6c1f-41fc-a1a2-08481b2d1579)



