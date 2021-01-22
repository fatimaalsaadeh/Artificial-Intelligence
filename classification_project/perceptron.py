# perceptron.py
# -------------

# Perceptron implementation
import util

PRINT = True


class PerceptronClassifier:
    """
    Perceptron classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter()  # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels);
        self.weights == weights;

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector a values).
        """

        self.features = trainingData[0].keys()  # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING

        for iteration in range(self.max_iterations):
            print("Starting iteration ", iteration, "...")
            for i in range(len(trainingData)):
                # "*** YOUR CODE HERE ***"
                yPrime = ""
                maxScore = 0

                for t in set(trainingLabels):
                    curScore = 0

                    for key, value in trainingData[i].items():
                        curScore += self.weights[t][key] * value

                    if curScore >= maxScore:
                        maxScore = curScore
                        yPrime = t

                if yPrime != trainingLabels[i]:
                    self.weights[trainingLabels[i]] = self.weights[trainingLabels[i]].__add__(trainingData[i])
                    self.weights[yPrime] = self.weights[yPrime].__sub__(trainingData[i])

            # util.raiseNotDefined()

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses

    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        featuresWeights = []

        "*** YOUR CODE HERE ***"

        sortedFeatures = sorted(self.weights[label], key=self.weights[label].__getitem__, reverse=True)

        x = 0
        for key in sortedFeatures:

            featuresWeights.append(key)
            # values.append(sortedFeatures.get(key))
            x += 1
            if x >= 100:
                break

        return featuresWeights
