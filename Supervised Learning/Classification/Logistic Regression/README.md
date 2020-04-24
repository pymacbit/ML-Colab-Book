## Logistic_Regression
- Don’t get confused by its name! It is a classification not a regression algorithm. It is used to estimate discrete values ( Binary values like 0/1, yes/no, true/false ) based on given set of independent variable(s).
- Linear regression predictions are continuous values (i.e., rainfall in cm), 
logistic regression predictions are discrete/categorical values (i.e., whether a student passed/failed) after applying a transformation function.
Hence, it is also known as **logit regression**.
- Logistic regression is best suited for binary classification: data sets where y = 0 or 1, where 1 denotes the default class. For example, in predicting whether an event will occur or not, there are only two possibilities: that it occurs (which we denote as 1) or that it does not (0). So if we were predicting whether a patient was sick, we would label sick patients using the value of 1 in our data set.
- Logistic regression is named after the transformation function it uses, which is called the logistic function

       h(x)= 1/ (1 + ex) 

- In logistic regression, the output takes the form of probabilities of the default class (unlike linear regression, where the output is directly produced). As it is a probability, the output lies in the range of 0-1. So, for example, if we’re trying to predict whether patients are sick, we already know that sick patients are denoted as 1, so if our algorithm assigns the score of 0.98 to a patient, it thinks that patient is quite likely to be sick.

  ![alt text](https://drive.google.com/uc?id=1DTRbSE5OxMlmyUrHnB_KVarVH5593Qjb)
