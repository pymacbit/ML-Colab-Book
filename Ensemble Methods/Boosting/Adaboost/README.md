## Boosting with AdaBoost

Adaboost stands for Adaptive Boosting. Bagging is a parallel ensemble because each model is built independently. On the other hand, boosting is a sequential ensemble where each model is built based on correcting the misclassifications of the previous model.

Bagging mostly involves ‘simple voting’, where each classifier votes to obtain a final outcome– one that is determined by the majority of the parallel models; boosting involves ‘weighted voting’, where each classifier votes to obtain a final outcome which is determined by the majority– but the sequential models were built by assigning greater weights to misclassified instances of the previous models.

Adaboost
Figure 9: Adaboost for a decision tree. Source

In Figure 9, steps 1, 2, 3 involve a weak learner called a decision stump (a 1-level decision tree making a prediction based on the value of only 1 input feature; a decision tree with its root immediately connected to its leaves).

The process of constructing weak learners continues until a user-defined number of weak learners has been constructed or until there is no further improvement while training. Step 4 combines the 3 decision stumps of the previous models (and thus has 3 splitting rules in the decision tree).

First, start with one decision tree stump to make a decision on one input variable.

The size of the data points show that we have applied equal weights to classify them as a circle or triangle. The decision stump has generated a horizontal line in the top half to classify these points. We can see that there are two circles incorrectly predicted as triangles. Hence, we will assign higher weights to these two circles and apply another decision stump.

Second, move to another decision tree stump to make a decision on another input variable.

We observe that the size of the two misclassified circles from the previous step is larger than the remaining points. Now, the second decision stump will try to predict these two circles correctly.

As a result of assigning higher weights, these two circles have been correctly classified by the vertical line on the left. But this has now resulted in misclassifying the three circles at the top. Hence, we will assign higher weights to these three circles at the top and apply another decision stump.

Third, train another decision tree stump to make a decision on another input variable.

The three misclassified circles from the previous step are larger than the rest of the data points. Now, a vertical line to the right has been generated to classify the circles and triangles.

Fourth, Combine the decision stumps.

We have combined the separators from the 3 previous models and observe that the complex rule from this model classifies data points correctly as compared to any of the individual weak learners.
