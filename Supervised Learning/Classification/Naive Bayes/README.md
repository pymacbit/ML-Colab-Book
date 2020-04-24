## Naive Bayes

- To calculate the probability that an event will occur, given that another event has already occurred, we use Bayes’s Theorem. To calculate the probability of hypothesis(h) being true, given our prior knowledge(d), we use Bayes’s Theorem as follows:

                P(h|d)= (P(d|h) P(h)) / P(d)

where:

                 - P(h|d) = Posterior probability. The probability of hypothesis h being true, given the data d, where     P(h|d)= P(d1| h) P(d2| h)….P(dn| h) P(d)
                 - P(d|h) = Likelihood. The probability of data d given that the hypothesis h was true.
                 - P(h)   = Class prior probability. The probability of hypothesis h being true (irrespective of the data)
                 - P(d)   = Predictor prior probability. Probability of the data (irrespective of the hypothesis)

- This algorithm is called ‘naive’ because it assumes that all the variables are independent of each other, which is a naive assumption to make in real-world examples.


 ![alt text](https://www.dataquest.io/wp-content/uploads/2019/01/Naive-Bayes.png)

Figure: Using Naive Bayes to predict the status of ‘play’ using the variable ‘weather’.

Using Figure above as an example, what is the outcome if weather = ‘sunny’?

To determine the outcome play = ‘yes’ or ‘no’ given the value of variable weather = ‘sunny’, calculate P(yes|sunny) and P(no|sunny) and choose the outcome with higher probability.

->P(yes|sunny)= (P(sunny|yes) * P(yes)) / P(sunny) = (3/9 * 9/14 ) / (5/14) = 0.60

-> P(no|sunny)= (P(sunny|no) * P(no)) / P(sunny) = (2/5 * 5/14 ) / (5/14) = 0.40

Thus, if the weather = ‘sunny’, the outcome is play = ‘yes’.

### Some of real world examples are:

- To mark an email as spam or not spam
- Classify a news article about technology, politics, or sports
- Check a piece of text expressing positive emotions, or negative emotions?
- Used for face recognition software.
