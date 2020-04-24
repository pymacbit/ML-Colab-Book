# **Reinforcement Learning**

- Reinforcement learning is a type of machine learning in which a computer learns to perform a task through repeated trial-and-error interactions with a dynamic environment. This learning approach enables the computer to make a series of decisions that maximize a reward metric for the task without human intervention and without being explicitly programmed to achieve the task. 

 ![alt text](https://drive.google.com/uc?id=17Fa-1lDG84Gr-oxoEfS8Kva2f0Rebxbc)


- Using this algorithm, the machine is trained to make specific decisions. It works this way: the machine is exposed to an environment where it trains itself continually using trial and error. This machine learns from past experience and tries to capture the best possible knowledge to make accurate business decisions.

- Reinforcement learning is a type of machine learning algorithm that allows an agent to decide the best next action based on its current state by learning behaviors that will maximize a reward.

- Reinforcement algorithms usually learn optimal actions through trial and error. Imagine, for example, a video game in which the player needs to move to certain places at certain times to earn points. A reinforcement algorithm playing that game would start by moving randomly but, over time through trial and error, it would learn where and when it needed to move the in-game character to maximize its point total. 

## **Reinforcement Learning Workflow**

 ![alt text](https://drive.google.com/uc?id=1CALKqpX27t0SjE4fS633oxEJV6yYCGkf)


**1. Create the Environment**

- First you need to define the environment within which the agent operates, including the interface between agent and environment. The environment can be either a simulation model or a real physical system. Simulated environments are usually a good first step since they are safer (real hardware is expensive!) and allow experimentation.

**2. Define the Reward**

- Next, specify the reward signal that the agent uses to measure its performance against the task goals and how this signal is calculated from the environment. Reward shaping can be tricky and may require a few iterations to get right.

**3. Create the Agent**

In this step, you create the agent. The agent consists of the policy and the training algorithm, so you need to: 

              a. Choose a way to represent the policy (e.g., using neural networks or look-up tables). 

              b. Select the appropriate training algorithm. Different representations are often tied to specific categories of training algorithms, but in general, most modern algorithms rely on neural networks because they are good candidates for large state/action spaces and complex problems.

**4. Train and Validate the Agent**

- Set up training options (e.g., stopping criteria) and train the agent to tune the policy. Make sure to validate the trained policy after training ends. Training can take anywhere from minutes to days depending on the application. For complex applications, parallelizing training on multiple CPUs, GPUs, and computer clusters will speed things up.

**5. Deploy the Policy**

- Deploy the trained policy representation using, for example, generated C/C++ or CUDA code. No need to worry about agents and training algorithms at this point—the policy is a standalone decision-making system!

- Training an agent using reinforcement learning is an iterative process. Decisions and results in later stages can require you to return to an earlier stage in the learning workflow. For example, if the training process does not converge on an optimal policy within a reasonable amount of time, you may have to update any of the following before retraining the agent:

- Training settings
- Learning algorithm configuration
- Policy representation
- Reward signal definition
- Action and observation signals
- Environment dynamics