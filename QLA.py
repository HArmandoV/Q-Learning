import numpy as np
import pandas as pd
import sys
import time
import tkinter as tk


class QLearning:
    def __init__(self, actions, learning=0.1, valueReward=0.9, pMovement=0.9):
        self.actions = actions  # a list
        self.learn = learning
        self.gamma = valueReward #Value of the future rewards. Way with the best rewards.
        self.epsilon = pMovement #Probability of an agent do a movement
        self.qTable = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def nextAction(self, observation):
        self.newState(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            stateAction = self.qTable.loc[observation, :]
            #print(stateAction)
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(stateAction[stateAction == np.max(stateAction)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def training(self, s, a, r, nextState):

        self.newState(nextState)
        qPredict = self.qTable.loc[s, a]
        
        if nextState != 'terminal':
            qTarget = r + self.gamma * self.qTable.loc[nextState, :].max()  # next state is not terminal
            #print(self.qTable)
        else:
            qTarget = r  # next state is terminal
        self.qTable.loc[s, a] += self.learn * (qTarget - qPredict)  # update

    def newState(self, state):
        if state not in self.qTable.index:
            # append new state to q table
            self.qTable = self.qTable.append(pd.Series([0]*len(self.actions), index=self.qTable.columns, name=state,))
#End QLearning

unit = 40   # pixels
gameH = 5  # grid height
gameW = 5  # grid width

class game(tk.Tk, object):
    def __init__(self):
        super(game, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('Q-Learning Algorithm')
        self.geometry('{0}x{1}'.format(gameH * unit, gameH * unit))
        self.buildGame()

    def buildGame(self):
        self.canvas = tk.Canvas(self, bg='beige', height=gameH * unit, width=gameW * unit)

        # create grids
        for c in range(0, gameW * unit, unit):
            x0, y0, x1, y1 = c, 0, c, gameH * unit
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, gameH * unit, unit):
            x0, y0, x1, y1 = 0, r, gameW * unit, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # blackSquare
        blackSquare1_center = origin + np.array([unit * 2, unit])
        self.blackSquare1 = self.canvas.create_rectangle(blackSquare1_center[0] - 15, blackSquare1_center[1] - 15, blackSquare1_center[0] + 15, blackSquare1_center[1] + 15, fill='black')
        # blackSquare
        blackSquare2_center = origin + np.array([unit * 2, unit *2])
        self.blackSquare2 = self.canvas.create_rectangle(blackSquare2_center[0] - 15, blackSquare2_center[1] - 15, blackSquare2_center[0] + 15, blackSquare2_center[1] + 15, fill='black')

        # create finalState
        finalState_center = origin + (unit * 4, unit * 4) 
        self.finalState = self.canvas.create_rectangle(finalState_center[0] - 15, finalState_center[1] - 15, finalState_center[0] + 15, finalState_center[1] + 15, fill='violet')

        # create rectangle
        self.rect = self.canvas.create_rectangle(origin[0] - 15, origin[1] - 15, origin[0] + 15, origin[1] + 15, fill='brown')

        # pack all
        self.canvas.pack()

    def again(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(origin[0] - 15, origin[1] - 15, origin[0] + 15, origin[1] + 15, fill='brown')
        return self.canvas.coords(self.rect)

    def move(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > unit:
                base_action[1] -= unit
        elif action == 1:   # down
            if s[1] < (gameH - 1) * unit:
                base_action[1] += unit
        elif action == 2:   # right
            if s[0] < (gameW - 1) * unit:
                base_action[0] += unit
        elif action == 3:   # left
            if s[0] > unit:
                base_action[0] -= unit

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        nextState = self.canvas.coords(self.rect)  # next state

        # reward function
        if nextState == self.canvas.coords(self.finalState):
            reward = 100
            done = True
            nextState = 'terminal'
        elif nextState in [self.canvas.coords(self.blackSquare1), self.canvas.coords(self.blackSquare2)]:
            reward = -1
            done = True
            nextState = 'terminal'
        else:
            reward = 0
            done = False
        return nextState, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    for i in range(10):
        se = env.again()
        while True:
            env.render()
            a = 1
            se, r, done = env.move(a)
            if done:
                break
#End game


def update():
    for episode in range(30):
        # initial observation
        observation = env.again()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.nextAction(str(observation))

            # RL take action and get next observation and reward
            observation, reward, done = env.move(action)

            # RL training from this transition
            RL.training(str(observation), action, reward, str(observation))

            # swap observation
            observation = observation

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = game()
    RL = QLearning(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
