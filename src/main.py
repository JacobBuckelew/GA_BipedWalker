#############################################################################################################################
# Imports

import gym
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from math import sqrt

# used for saving results of experiments
import timeit
import json

#############################################################################################################################


# normalized xavier normalization is commonly used for initializing weights for networks using sigmoid or tanh
def normalized_xavier(inputs,outputs,mult):

    xavier = mult*np.sqrt(6.0/(inputs+outputs))
    # number of nodes in the previous layer
    #n = inputs
    # number of nodes in the next layer
    #m = outputs
    # calculate the range for the weights
    #lower, upper = -(sqrt(6.0) / sqrt(n + m)), (sqrt(6.0) / sqrt(n + m))
    #return np.random.uniform(lower, upper, size=(n,m))
    #scaled = lower + numbers * (upper - lower)
    return np.random.uniform(-1 * xavier,xavier,size=(inputs,outputs))


# softmax will convert the fitness scores into probability values. Temperature controls the randomness of output distribution
def softmax(fitnesses,temperature):
    
    e = np.exp(np.array(fitnesses)/temperature)
    return e/e.sum()


# Our Agent is represented by an MLP consisting of 2 layers
# Use xavier uniform initialization for each of the layers including a bias
# Our selected activation function for each layer will be tanh

class Agent(object):

    # Initialize the parameters of the model
    def __init__(self, inputs, hidden, outputs, mutate, mult):
        self.mutate= mutate
        self.mult= mult
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.network = {'Layer 1' : normalized_xavier(inputs, hidden,mult),
                        'Bias 1'  : np.zeros((1,hidden)),
                        'Layer 2' : normalized_xavier(hidden, outputs,mult),
                        'Bias 2'  : np.zeros((1,outputs))}
                        
    

    # This method overloads the + operator for the purpose of reproduction
    # between agents. 
    def __add__(self, mate):
        child = Agent(self.inputs, self.hidden, self.outputs, self.mutate, self.mult)
        
        for key in child.network:
            #print("key", key)
            mask = np.random.choice([0,1],size=child.network[key].shape,p=[.5,.5])
            #print("mask", mask)
            #print(mask.shape)
            random = normalized_xavier(mask.shape[0],mask.shape[1], self.mult)
            #print("random", random)
            #print(random.shape)
            #print("self:", self.network[key])
            #print("mate:", mate.network[key])
            child.network[key] = np.where(mask==1,self.network[key],mate.network[key])
            #print("new key:", child.network[key])
            mask = np.random.choice([0,1],size=child.network[key].shape,p=[1-self.mutate,self.mutate])
            #print("mask:", mask)
            #print(mask.shape)
            child.network[key] = np.where(mask==1,child.network[key]+random,child.network[key])
            #print(child.network[key])
        return child



    # Act is essentially the forward pass through the network
    # uses the current state to determine the action the agent will take
    def act(self, state):       
        if(state.shape[0] != 1):
            state = state.reshape(1,-1)
        net = self.network
        # take state as input into first layer
        layer_one = np.tanh(np.matmul(state,net['Layer 1']) + net['Bias 1'])
        # pass first layers output into next layer as input
        layer_two = np.tanh(np.matmul(layer_one, net['Layer 2']) + net['Bias 2'])
        return layer_two[0]
    


# run a single trial for an agent using run_trial()
def run_trial(env,agent,verbose=False):
    EPISODES = 4
    # average across 4 episodes
    rewards = []
    for _ in range(EPISODES):
        state = env.reset()
        if verbose: env.render(mode="human")
        total = 0
        done = False
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            if verbose: env.render()
            total += reward
        rewards.append(total)
    return sum(rewards)/EPISODES


# get next generation of agents using next_generation()
def next_generation(env,population,fitnesses,temperature):

    fitnesses, population =  zip(*sorted(zip(fitnesses,population),reverse=True))

    # elitism using top 25% of individuals
    children = list(population[:int(len(population)/4)])

    # choose individuals using their softmax scores
    parents = list(np.random.choice(population,size=2*(len(population)-len(children)),p=softmax(fitnesses,temperature)))

    # crossover traits
    children = children + [parents[i]+parents[i+1] for i in range(0,len(parents)-1,2)]

    fitnesses = [run_trial(env,agent) for agent in children]

    return children,fitnesses


# Main function for running the training process

def main():
    
    # initialize the bipedalwalker-v2 environment
    env = gym.make('BipedalWalker-v2')

    # for replication purposes
    np.random.seed(4)
    env.seed(4)
    
    
    # see the observation and action spaces
    obs_space = env.observation_space.shape
    action_space = env.action_space.shape


    # Define some of the parameters for the MLP
    # We need to track features, number of hidden layers, actions, and a multiplier
    features = obs_space[0]
    hidden_layers = 512
    mult = 10
    actions = action_space[0]
    
    
    # Set parameters for the genetic algorithm
    # Population size, mutation chance, and a softmax_temp
    # temp is useful for softmax specifically since it helps control randomness
    population_size = 80
    mutate= 0.11
    temp = 20.0
    win_condition = 200
    
    # Timer for timing the training process
    start_time = timeit.default_timer()
    generations = 100
    population = [Agent(features,hidden_layers,actions,mutate,mult) for i in list(range(population_size))]
    fitnesses = [run_trial(env,agent) for agent in population]
    avg_fitnesses = []
    max_fitnesses = []
    first_gen = 0
    best = [deepcopy(population[np.argmax(fitnesses)])]
    best_fitness = np.max(fitnesses)
    best_generation = 0
    for generation in list(range(generations)):
        population,fitnesses = next_generation(env,population, fitnesses,temp)
        best.append(deepcopy(population[np.argmax(fitnesses)]))
        print("Generation:",generation,"fitness:",np.max(fitnesses))
        if(np.max(fitnesses) >= win_condition):
            first_gen = generation
        if(np.max(fitnesses) > best_fitness):
            best_generation = generation
            best_fitness = np.max(fitnesses)
        avg_fitnesses.append(np.mean(fitnesses))
        max_fitnesses.append(np.max(fitnesses))

    end_time =  timeit.default_timer()

    # Store results of training
    result_dict = {}

    result_dict["train_time"] = (end_time - start_time)/60.0
    result_dict["first_gen"] = first_gen
    result_dict["best_generation"] = best_generation
    result_dict["overall_bestfitness"] = best_fitness
    result_dict["final_bestfitness"] = max_fitnesses[-1]
    result_dict["final_avgfitness"] = avg_fitnesses[-1]

    with open("/Users/jacobbuckelew/Documents/CS4260/GA_BipedWalker/results/metrics.json", 'w') as file:
            json.dump(result_dict, file)



    # plotting max and avg fitnesses across the generations
    fig = plt.figure(figsize=(14,10))
    x = len(avg_fitnesses)
    plt.plot(avg_fitnesses, label="Average Fitness", linewidth='5.0')
    plt.plot(max_fitnesses, label="Max Fitness", linewidth='5.0', ls='--')
    plt.xlabel("Generation", fontsize='32')
    plt.ylabel("Fitness", fontsize='32')
    plt.legend(fontsize='28')
    plt.yticks(fontsize=28)
    plt.xticks(fontsize=28)
    plt.savefig("/Users/jacobbuckelew/Documents/CS4260/GA_BipedWalker/results/fitness_curve.png", dpi=500)

    # Create recordings of the agents every 3 episodes
    env = gym.wrappers.Monitor(env,'./video',force=True,video_callable=lambda episode_id: episode_id%4==0)  
    for agent in best:
        print("Recording")
        run_trial(env,agent)
    env.close()
    
if __name__ == '__main__':
    main()