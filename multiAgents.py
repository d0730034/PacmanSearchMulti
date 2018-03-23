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
import random, util, sys

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

        "*** YOUR CODE HERE ***"
	#getting the food list from the newFood
	new_food_list = newFood.asList()
	#if there is no more food in the food list then return end of the function
        if len(new_food_list) == 0:
          return successorGameState.getScore()

	#basic set up for minimum distance of food and ghost
	minimum_food_distance = sys.maxint
	minimum_ghost_distance = sys.maxint
	new_ghost_list = []

	#calculate the minimum distance of food from the new_food_list
	for food in new_food_list:
	    if manhattanDistance(newPos, food) < minimum_food_distance:
	        minimum_food_distance = manhattanDistance(newPos, food)

	#load the ghost state information and build up the new_ghost_list
	for ghost in newGhostStates:
		new_ghost_list.append(ghost.getPosition())

	#calculate the minimum distance of ghost from the new_ghost_list
	for ghost in new_ghost_list:
	    if manhattanDistance(newPos, ghost) < minimum_ghost_distance:
	        minimum_ghost_distance = manhattanDistance(newPos, ghost)

	chosen = 0
	
	if minimum_ghost_distance > 2:
	    chosen = 1 / (len(new_food_list) + 0.01 * minimum_food_distance)
	return chosen

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

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
        """
        "*** YOUR CODE HERE ***"
	#information: https://stackoverflow.com/questions/7604966/maximum-and-minimum-values-for-ints

        def min_layer(gameState, depth, agent_id):
	    #getting the possible move action
	    legal_action_list = gameState.getLegalActions(agent_id)

	    #set up the result score and result action for return
	    result_score = sys.maxint #information: https://stackoverflow.com/questions/7604966/maximum-and-minimum-values-for-ints
	    result_action = None

	    #return the result if there is no possible move
	    if len(legal_action_list) == 0:
	        return (self.evaluationFunction(gameState), None)

	    #find the best possible action movement
	    for legal_action in legal_action_list:
		current_state = gameState.generateSuccessor(agent_id, legal_action)
		if agent_id == gameState.getNumAgents() - 1:
		    current_score = max_layer(current_state, depth + 1)[0]
		else:
		    current_score = min_layer(current_state, depth, agent_id + 1)[0]
		#compare and get the best score and action
		if (current_score < result_score):
		    result_score = current_score
		    result_action = legal_action
	    return(result_score, result_action)


	def max_layer(gameState, depth):
	    #getting the possible move action
	    legal_action_list = gameState.getLegalActions(0) #0 stand for pacman is always agent 0
	    
	    #return the result if there is no possible move or reach the depth
	    if len(legal_action_list) == 0 or depth == self.depth:
	        return (self.evaluationFunction(gameState), None)

	    #set up the result score and result action for return
	    result_score = -sys.maxint - 1 #information: https://stackoverflow.com/questions/7604966/maximum-and-minimum-values-for-ints
	    result_action = None

	    #find the best result and compare the all possible action from the legal_action_list
	    for legal_action in legal_action_list:
	        current_state = gameState.generateSuccessor(0, legal_action)
		current_score = min_layer(current_state, depth, 1)[0]
		#compare and get the best score and action
		if (current_score > result_score):
		    result_score = current_score
		    result_action = legal_action
	    return (result_score, result_action)


        return max_layer(gameState, 0)[1]

        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def min_layer(gameState, depth, agent_id, alpha, beta):
	    #getting the possible move action
	    legal_action_list = gameState.getLegalActions(agent_id)

	    #set up the result score and result action for return
	    result_score = sys.maxint #information: https://stackoverflow.com/questions/7604966/maximum-and-minimum-values-for-ints
	    result_action = None

	    #return the result if there is no possible move
	    if len(legal_action_list) == 0:
	        return (self.evaluationFunction(gameState), None)

	    #find the best possible action movement
	    for legal_action in legal_action_list:
		#according to the alpha-beta algorithm in min_layer if beta smaller than alpha then return
                if beta < alpha:
		    return (result_score, result_action)

		current_state = gameState.generateSuccessor(agent_id, legal_action)
		if agent_id == gameState.getNumAgents() - 1:
		    current_score = max_layer(current_state, depth + 1, alpha, beta)[0]
		else:
		    current_score = min_layer(current_state, depth, agent_id + 1, alpha, beta)[0]
		#compare and get the best score and action
		if (current_score < result_score):
		    result_score = current_score
		    result_action = legal_action
		if beta > current_score:
		    beta = current_score
	    return(result_score, result_action)


	def max_layer(gameState, depth, alpha, beta):
	    #getting the possible move action
	    legal_action_list = gameState.getLegalActions(0) #0 stand for pacman is always agent 0
	    
	    #return the result if there is no possible move or reach the depth
	    if len(legal_action_list) == 0 or depth == self.depth:
	        return (self.evaluationFunction(gameState), None)

	    #set up the result score and result action for return
	    result_score = -sys.maxint - 1 #information: https://stackoverflow.com/questions/7604966/maximum-and-minimum-values-for-ints
	    result_action = None

	    #find the best result and compare the all possible action from the legal_action_list
	    for legal_action in legal_action_list:
		if beta < alpha:
		    return(result_score, result_action)
	        current_state = gameState.generateSuccessor(0, legal_action)
		current_score = min_layer(current_state, depth, 1, alpha, beta)[0]
		#compare and get the best score and action
		if (current_score > result_score):
		    result_score = current_score
		    result_action = legal_action
		if alpha < current_score:
		    alpha = current_score
	    return (result_score, result_action)

        return max_layer(gameState, 0, -sys.maxint - 1, sys.maxint)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
	def expect_layer(gameState, depth, agent_id):
	    #getting the possible move action
	    legal_action_list = gameState.getLegalActions(agent_id)

	    #set up the result score and result action for return
	    result_score = 0
	    result_action = None

	    #return the result if there is no possible move
	    if len(legal_action_list) == 0:
	        return (self.evaluationFunction(gameState), None)

	    #find the best possible action movement
	    for legal_action in legal_action_list:
		current_state = gameState.generateSuccessor(agent_id, legal_action)
		if agent_id == gameState.getNumAgents() - 1:
		    current_score = max_layer(current_state, depth + 1)[0]
		else:
		    current_score = expect_layer(current_state, depth, agent_id + 1)[0]
		result_score = result_score + (current_score/len(legal_action_list))
	    return(result_score, result_action)


	def max_layer(gameState, depth):
	    #getting the possible move action
	    legal_action_list = gameState.getLegalActions(0) #0 stand for pacman is always agent 0
	    
	    #return the result if there is no possible move or reach the depth
	    if len(legal_action_list) == 0 or depth == self.depth:
	        return (self.evaluationFunction(gameState), None)

	    #set up the result score and result action for return
	    result_score = -sys.maxint - 1 #information: https://stackoverflow.com/questions/7604966/maximum-and-minimum-values-for-ints
	    result_action = None

	    #find the best result and compare the all possible action from the legal_action_list
	    for legal_action in legal_action_list:
	        current_state = gameState.generateSuccessor(0, legal_action)
		current_score = expect_layer(current_state, depth, 1)[0]
		#compare and get the best score and action
		if (current_score > result_score):
		    result_score = current_score
		    result_action = legal_action
	    return (result_score, result_action)


        return max_layer(gameState, 0)[1]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #copy and past from the previous evaluationfunction given
    #Replace SuccessorGameState to currentGamState as given
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


    result = 0.0 + currentGameState.getScore()

    minimum_food_distance = sys.maxint
    total_pacman_distance = 1

    new_food_list = newFood.asList()

    for food in new_food_list:
        total_pacman_distance = total_pacman_distance + manhattanDistance(newPos, food)
        if manhattanDistance(newPos, food) < minimum_food_distance:
          minimum_food_distance = manhattanDistance(newPos, food)
      
    result = result + (1 / (minimum_food_distance ** 2)  + 1 / total_pacman_distance)
            
    for ghost in newGhostStates:
        if manhattanDistance(newPos, ghost.getPosition()) <= 5 and manhattanDistance(newPos, ghost.getPosition()) > ghost.scaredTimer:
            result = result - (1 / (manhattanDistance(newPos, ghost.getPosition()) ** 2)) - 0.8
    return result

    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

