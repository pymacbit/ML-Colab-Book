## **Polynomial Regression**

- A regression equation is a polynomial regression equation if the power of independent variable is more than 1. The equation below represents a polynomial equation:

            y=a+b*x^2

     ![alt text](https://drive.google.com/uc?id=1KUBsFUslsY7hj1Fzm0e0mhkILvwijl4j)

- In this regression technique, the best fit line is not a straight line. It is rather a curve that fits into the data points.

- Polynomial Regression is a form of linear regression in which the relationship between the independent variable x and dependent variable y is modeled as an nth degree polynomial. Polynomial regression fits a nonlinear relationship between the value of x and the corresponding conditional mean of y, denoted E(y |x)

- The basic goal of regression analysis is to model the expected value of a dependent variable y in terms of the value of an independent variable x. In simple regression, we used following equation –

          y = a + bx + e

          Here y is dependent variable, a is y intercept, b is the slope and e is the error rate.

- In many cases, this linear model will not work out For example if we analyzing the production of chemical synthesis in terms of temperature at which the synthesis take place in such cases we use quadratic model

          y = a + b1x + b2^2 + e
          Here y is dependent variable on x, a is y intercept and e is the error rate.

- In general, we can model it for nth value.

          y = a + b1x + b2x^2 +....+ bnx^n

- Since regression function is linear in terms of unknown variables, hence these models are linear from the point of estimation.

### **Why Polynomial Regression:**

- There are some relationships that a researcher will hypothesize is curvilinear. Clearly, such type of cases will include a polynomial term.
- Inspection of residuals. If we try to fit a linear model to curved data, a scatter plot of residuals (Y axis) on the predictor (X axis) will have patches of many positive residuals in the middle. Hence in such situation it is not appropriate.
- An assumption in usual multiple linear regression analysis is that all the independent variables are independent. In polynomial regression model, this assumption is not satisfied.

### **Important Points**:

- While there might be a temptation to fit a higher degree polynomial to get lower error, this can result in over-fitting. Always plot the relationships to see the fit and focus on making sure that the curve fits the nature of the problem. Here is an example of how plotting can help:

     ![alt text](https://drive.google.com/uc?id=1cMmax1QwPOJVRJ0Jzf4tbPvgMs45MHHd)

- Especially look out for curve towards the ends and see whether those shapes and trends make sense. Higher polynomials can end up producing wierd results on extrapolation.

### **Advantages of using Polynomial Regression:**

Polynomial provides the best approximation of the relationship between the dependent and independent variable.
A Broad range of function can be fit under it.
Polynomial basically fits a wide range of curvature.

### **Disadvantages of using Polynomial Regression**

The presence of one or two outliers in the data can seriously affect the results of the nonlinear analysis.
These are too sensitive to the outliers.
In addition, there are unfortunately fewer model validation tools for the detection of outliers in nonlinear regression than there are for linear regression.
