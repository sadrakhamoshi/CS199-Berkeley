# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        res = []
        for food in newFood.asList():
            res.append(1 / util.manhattanDistance(food, newPos))

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        return score if len(res) == 0 else score + max(res)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    action = None
    all_agent_num = 0

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        self.action = None
        self.all_agent_num = gameState.getNumAgents()
        minimax_value = self.MiniMaxFunc(gameState, 0, 0)
        return self.action
        util.raiseNotDefined()

    def MiniMaxFunc(self, gameState, turn, depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        agent_id = turn % self.all_agent_num
        if agent_id == 0:
            if depth >= self.depth:
                return self.evaluationFunction(gameState)
            else:
                return self.max_value(gameState, turn, depth)
        else:
            return self.min_value(gameState, turn, depth)

    def max_value(self, gameState, turn, depth):
        v = -9999999
        legal_action = gameState.getLegalActions(0)
        for action in legal_action:
            state = gameState.generateSuccessor(0, action)
            maximum = self.MiniMaxFunc(state, turn + 1, depth + 1)
            if maximum > v:
                v = maximum
                if turn == 0:
                    self.action = action
        return v

    def min_value(self, gameState, turn, depth):
        v = +9999999
        index = turn % self.all_agent_num
        legal_action = gameState.getLegalActions(index)
        for action in legal_action:
            state = gameState.generateSuccessor(index, action)
            v = min(v, self.MiniMaxFunc(state, turn + 1, depth))
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    action = None
    all_agent_num = 0

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        self.all_agent_num = gameState.getNumAgents()
        self.alpha_beta(gameState, 0, 0, -999999, +999999)
        return self.action
        util.raiseNotDefined()

    def alpha_beta(self, gameState, turn, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        agent_id = turn % self.all_agent_num
        if agent_id == 0:
            if depth >= self.depth:
                return self.evaluationFunction(gameState)
            else:
                return self.max_value(gameState, turn, depth, alpha, beta)
        else:
            return self.min_value(gameState, turn, depth, alpha, beta)

    def max_value(self, gameState, turn, depth, alpha, beta):
        v = -9999999
        legal_action = gameState.getLegalActions(0)
        for action in legal_action:
            state = gameState.generateSuccessor(0, action)
            maximum = self.alpha_beta(state, turn + 1, depth + 1, alpha, beta)
            if maximum > v:
                v = maximum
                if turn == 0:
                    self.action = action
            if v > beta:
                return v
            alpha = max(v, alpha)

        return v

    def min_value(self, gameState, turn, depth, alpha, beta):
        v = +9999999
        index = turn % self.all_agent_num
        legal_action = gameState.getLegalActions(index)
        for action in legal_action:
            state = gameState.generateSuccessor(index, action)
            v = min(v, self.alpha_beta(state, turn + 1, depth, alpha, beta))
            if v < alpha:
                return v
            beta = min(v, beta)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    action = None
    all_agent_num = 0

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        self.all_agent_num = gameState.getNumAgents()
        self.ExpectiMax(gameState, 0, 0)
        return self.action
        util.raiseNotDefined()

    def ExpectiMax(self, gameState, turn, depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        agent_id = turn % self.all_agent_num
        if agent_id == 0:
            if depth >= self.depth:
                return self.evaluationFunction(gameState)
            else:
                return self.max_value(gameState, turn, depth)
        else:
            return self.exp_value(gameState, turn, depth)

    def max_value(self, gameState, turn, depth):
        v = -9999999
        legal_action = gameState.getLegalActions(0)
        for action in legal_action:
            state = gameState.generateSuccessor(0, action)
            maximum = self.ExpectiMax(state, turn + 1, depth + 1)
            if maximum > v:
                v = maximum
                if turn == 0:
                    self.action = action
        return v

    def exp_value(self, gameState, turn, depth):
        v = 0
        index = turn % self.all_agent_num
        legal_action = gameState.getLegalActions(index)
        for action in legal_action:
            state = gameState.generateSuccessor(index, action)
            v += ((1 / len(legal_action)) * self.ExpectiMax(state, turn + 1, depth))
        return v


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacman_pos = currentGameState.getPacmanPosition()
    all_food = currentGameState.getFood().asList()
    score = currentGameState.getScore()
    ghost_states = currentGameState.getGhostStates()
    value = score
    for ghost in ghost_states:
        dist_ghost = util.manhattanDistance(pacman_pos, ghost.configuration.getPosition())
        if dist_ghost != 0:
            if ghost.scaredTimer > 0:
                value += 100 / dist_ghost
            else:
                value -= 10 / dist_ghost

    # dist to closet food
    dist_food = []
    for food in all_food:
        dist_food.append(1 / util.manhattanDistance(food, pacman_pos))
    if len(dist_food) > 0:
        value += max(dist_food) * 10

    return value
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
