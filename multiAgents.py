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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"
        # *** YOUR CODE HERE *** 反射式评估函数：鼓励靠近食物，远离非惊吓鬼，吃到食物加分
        if successorGameState.isWin():
            return float('inf')
        if successorGameState.isLose():
            return float('-inf')

        score = successorGameState.getScore()

        # 距离最近食物（越近越好）
        foodList = newFood.asList()
        if foodList:
            minFoodDist = min(manhattanDistance(newPos, f) for f in foodList)
            score += 10.0 / (1 + minFoodDist)

        # 处理鬼距离：未被惊吓的鬼需要远离，被惊吓的鬼可以追
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            dist = manhattanDistance(newPos, ghostPos)
            if scaredTime > 0:
                # 鬼被惊吓：靠近有利
                score += 2.0 / (1 + dist)
            else:
                # 鬼未惊吓：太近惩罚
                if dist <= 1:
                    score -= 10
                score -= 2.0 / (1 + dist)

        # 避免停留
        if action == Directions.STOP:
            score -= 3

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (Question 4)
    """

    def getAction(self, gameState: GameState):
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
        # *** YOUR CODE HERE *** Minimax 多智能体搜索：Pacman 最大化，鬼最小化，到达深度或终局用评估函数
        numAgents = gameState.getNumAgents()

        def isTerminal(state, depth):
            return depth == self.depth or state.isWin() or state.isLose()

        def value(state, depth, agentIndex):
            if isTerminal(state, depth):
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return maxValue(state, depth)
            else:
                return minValue(state, depth, agentIndex)

        def maxValue(state, depth):
            v = float('-inf')
            actions = state.getLegalActions(0)
            if not actions:
                return self.evaluationFunction(state)
            for action in actions:
                succ = state.generateSuccessor(0, action)
                v = max(v, value(succ, depth, 1))
            return v

        def minValue(state, depth, agentIndex):
            v = float('inf')
            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)
            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == numAgents:
                nextAgent = 0
                nextDepth = depth + 1
            for action in actions:
                succ = state.generateSuccessor(agentIndex, action)
                v = min(v, value(succ, nextDepth, nextAgent))
            return v

        bestScore = float('-inf')
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            score = value(succ, 0, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Qestion 8)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # *** YOUR CODE HERE *** Alpha-Beta 剪枝版 Minimax
        numAgents = gameState.getNumAgents()

        def isTerminal(state, depth):
            return depth == self.depth or state.isWin() or state.isLose()

        def value(state, depth, agentIndex, alpha, beta):
            if isTerminal(state, depth):
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return maxValue(state, depth, alpha, beta)
            else:
                return minValue(state, depth, agentIndex, alpha, beta)

        def maxValue(state, depth, alpha, beta):
            v = float('-inf')
            actions = state.getLegalActions(0)
            if not actions:
                return self.evaluationFunction(state)
            for action in actions:
                succ = state.generateSuccessor(0, action)
                v = max(v, value(succ, depth, 1, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(state, depth, agentIndex, alpha, beta):
            v = float('inf')
            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)
            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == numAgents:
                nextAgent = 0
                nextDepth = depth + 1
            for action in actions:
                succ = state.generateSuccessor(agentIndex, action)
                v = min(v, value(succ, nextDepth, nextAgent, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        bestScore = float('-inf')
        bestAction = Directions.STOP
        alpha, beta = float('-inf'), float('inf')
        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            score = value(succ, 0, 1, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            if bestScore > beta:
                return bestAction
            alpha = max(alpha, bestScore)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Question 0)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # *** YOUR CODE HERE *** Expectimax：鬼为期望节点（等概率选择动作）
        numAgents = gameState.getNumAgents()

        def isTerminal(state, depth):
            return depth == self.depth or state.isWin() or state.isLose()

        def value(state, depth, agentIndex):
            if isTerminal(state, depth):
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return maxValue(state, depth)
            else:
                return expValue(state, depth, agentIndex)

        def maxValue(state, depth):
            v = float('-inf')
            actions = state.getLegalActions(0)
            if not actions:
                return self.evaluationFunction(state)
            for action in actions:
                succ = state.generateSuccessor(0, action)
                v = max(v, value(succ, depth, 1))
            return v

        def expValue(state, depth, agentIndex):
            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)
            prob = 1.0 / len(actions)
            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == numAgents:
                nextAgent = 0
                nextDepth = depth + 1
            expected = 0.0
            for action in actions:
                succ = state.generateSuccessor(agentIndex, action)
                expected += prob * value(succ, nextDepth, nextAgent)
            return expected

        bestScore = float('-inf')
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            score = value(succ, 0, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Question 10).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # *** YOUR CODE HERE *** 综合评估：考虑当前分数、最近食物、剩余食物数、与鬼距离（惊吓与否）
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    score = currentGameState.getScore()
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    # 食物要素：越少越好；最近食物越近越好
    if food:
        minFoodDist = min(manhattanDistance(pacmanPos, f) for f in food)
        # 更强地鼓励靠近最近食物
        score += 22.0 / (1 + minFoodDist)
        # 更明显地惩罚剩余食物（推动尽快清屏）
        score -= 2.5 * len(food)

    # 鬼要素：未惊吓需要保持距离，被惊吓可以接近
    minUnsafeGhostDist = float('inf')
    for gs in ghosts:
        ghostPos = gs.getPosition()
        dist = manhattanDistance(pacmanPos, ghostPos)
        if gs.scaredTimer > 0:
            # 被惊吓的鬼：更积极地靠近
            score += 6.0 / (1 + dist)
            # 若马上可吃，给小额奖励
            if dist == 0:
                score += 20
        else:
            # 未惊吓的鬼：强力规避
            minUnsafeGhostDist = min(minUnsafeGhostDist, dist)
            if dist <= 1:
                score -= 35
            elif dist <= 2:
                score -= 20
            # 距离越近惩罚越大
            score -= 5.0 / (1 + dist)

    # 胶囊：剩余越少越好；如果附近存在未惊吓鬼，鼓励靠近胶囊
    if capsules:
        score -= 3.0 * len(capsules)
        minCapDist = min(manhattanDistance(pacmanPos, c) for c in capsules)
        # 当存在危险鬼时，鼓励朝最近胶囊移动
        if minUnsafeGhostDist < float('inf') and minUnsafeGhostDist <= 3:
            score += 10.0 / (1 + minCapDist)
        else:
            # 平时也略微鼓励靠近胶囊（便于后续处理鬼）
            score += 4.0 / (1 + minCapDist)

    # 避免停留（虽然 betterEvaluationFunction 不直接使用动作，但可倾向动态策略）
    return score

# Abbreviation
better = betterEvaluationFunction
