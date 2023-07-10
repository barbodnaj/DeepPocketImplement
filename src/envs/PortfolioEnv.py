import torch

class PortfolioEnv():
    
    def __init__(self,portfolioSize:int = 2,portfolioValue:int = 1000,buyCommission = 0.001,sellCommission = 0.001,maxSteps = 100):
        self.portfolioSize = portfolioSize + 1 # plus one for the cash balance

        self.numberOfSteps = 0 
        self.maxSteps = maxSteps
        self.states = []
        # Define action and observation spaces

        # self.action_space = spaces.Box(low=-1, high=1, shape=(self.portfolioSize,),dtype=torch.float32) 
        # self.observation_space = spaces.Box(low=0, high=1, shape=(self.portfolioSize,),dtype=torch.float32)

        # Start with only cash last index is cash value
        self.state = torch.zeros(self.portfolioSize, dtype=torch.float32)  
        self.state[self.portfolioSize-1] = torch.tensor(1,dtype=torch.float32)
        self.states.append(self.state)

        
        self.initPortfolioValue =torch.tensor(portfolioValue,dtype=torch.float32)
        self.portfolioValue =torch.tensor(portfolioValue,dtype=torch.float32)

        self.bC =torch.tensor(buyCommission,dtype=torch.float32)
        self.sC = torch.tensor(sellCommission,dtype=torch.float32)

        self.totalReward = 0

    def step(self, action):
        # assert self.action_space.contains(action), "Invalid action"

        self.lastState = self.state.clone()
        # Update the state based on the action
        self.state = torch.clamp(self.state + action, 0, 1)

        # Normalize the state tensor to ensure the sum is 1
        self.state /= torch.sum(self.state)

        self.states.append(self.state)

        # calculating the portfolio value
        stateDifference = self.state-self.lastState
        commission = torch.where(stateDifference > 0, self.bC, self.sC).T
        valueOfEachAsset = (self.portfolioValue*self.state) - torch.matmul(torch.pow(self.portfolioValue*stateDifference,2),commission,dtype=torch.float32)
        valueOfEachAsset[self.portfolioSize-1] = self.state[self.portfolioSize-1]*self.portfolioValue
        
        self.portfolioLastValue = self.portfolioValue.clone()
        self.portfolioValue = torch.sum(valueOfEachAsset)

        # Define the reward function (you can modify this to suit your needs)
        reward = self.reward()
        self.totalReward += reward

        self.numberOfSteps = 1+self.numberOfSteps
        if(self.numberOfSteps>= self.maxSteps):
            done = 1
        else:
            done = 0

        info:dict ={
            "numberOfSteps":self.numberOfSteps,
            "reward":reward,
            "totalReward":self.totalReward,
            "valueOfEachAsset":valueOfEachAsset,
            "portfolioValue":self.portfolioValue,
            "state":self.state,
        } 
        # Return the next state,next value, reward, termination flag, and additional information
        return self.state.clone(),self.portfolioValue.clone(), reward, done, info

    def reset(self):
        self.states = []

        # Reset the state to the initial configuration
        self.state = torch.zeros(self.portfolioSize, dtype=torch.float32)  
        self.state[self.portfolioSize-1] = 1
        self.states.append(self.state)

        self.portfolioValue = self.initPortfolioValue
        
        self.numberOfSteps = 0
        self.totalReward = 0
 

        return self.state.clone()
    
    def reward(self):
        return torch.log(self.portfolioValue/self.portfolioLastValue)
        


        

