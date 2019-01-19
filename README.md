[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Navigation in Banana Maze with Q-learning based Agent

### Introduction

This project provides trainable agent (via Q-learning) navigating in a large, square world. On top of framework for training, we provide also already trained agents with detailed report addressing details of training algorithms and interesting next steps possible on the project.  

![Trained Agent][image1]

### Project details

The goal of agent in environment is to collect as many yellow bananas as possible while avoiding blue ones.  
Given the information from environment, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The task is episodic, and in order to solve the environment, agent must get an **average score of +13 over 100 consecutive episodes**.

Agent percents environment via state input information. These can be structured (default setup in project) and unstructured
#### Structured input
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.

#### Unstructured input

The state space is an 84 x 84 RGB image, corresponding to the agent's first-person view.  
This environment is not currently used in package. You'll need to download a new Unity environment. You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)

Then, extract the file in the `bin/` folder and use in the same manner as original environment in `bin/unity_banana_maze`

### Getting Started
 - ```python setup.py develop``` - This installs package with all requirements in develop mode for interactive updates.
 - ```install ``` [```Unity ML-Agents```](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) - Unity plugin that enables games and simulations to serve as environments for training intelligent agents, Unity environment is already compiled in project.
 - ```python setup.py test -a '-v  tests/'``` - Running all tests.

### Structure of package

 - ```qrl_navigation``` - Package core with implementation of agent, environment, replay buffers etc.
 - ```tests```
 - ```bin``` - Compiled Unity environment ready for Linux operating systems.
 - ```experiments``` - Couple of experiments including trained dueling Q networks and weighted replay buffers. More details in ```notebooks/report.ipynb```
 - ```notebooks``` - Insight to usage of package and description of results.
   - ```train.ipynb``` - Example of agent training.
   - ```report.ipynb``` - Analysis of experiments and proposal of next steps.
   - ```environment_introduction.ipynb``` - Example of environment interaction.   

### Instructions

Check content of ```notebooks``` , more details in **Structure of package** section. There are examples of training as well as current results analysis and proposal of next steps.
