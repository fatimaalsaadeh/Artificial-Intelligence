# mlp.py
# -------------

# mlp implementation
import util
import numpy as np

PRINT = True


class MLPClassifier:
    """
  mlp classifier
  """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mlp"
        self.max_iterations = max_iterations

    def sigmoid(self, x, d):
        if (d == True):
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        tData = []
        vData = []
        tLabels = np.zeros((len(trainingLabels), 10))

        # Converting the training and validation data to the input format
        for i in range(len(trainingData)):
            sample = trainingData[i].items()
            if i < len(validationData):
                validation = validationData[i].items()

            temp1 = []
            temp2 = []
            tLabels[i][trainingLabels[i]] = 1
            for j in range(len(sample)):
                # Only appending the feature at that point
                # Number of features should be the same for all training samples
                temp1.append(sample[j][1])
                if i < len(validationData):
                    temp2.append(validation[j][1])
            sample = np.asarray(temp1)
            sample = sample.flatten()
            tData.append(sample)
            if i < len(validationData):
                validation = np.asarray(temp2)
                validation = validation.flatten()
                vData.append(validation)
        tData = np.asarray(tData)

        k = 150
        # 150 works decent for digits
        features = len(tData[0])
        self.l0 = np.random.random((features, k)) - 0.5
        self.l1 = np.random.random((k, 10)) - 0.5
        wl0 = self.l0
        wl1 = self.l1
        learning = 0.02

        for iteration in range(self.max_iterations):
            print ("Starting iteration ", iteration, "...")
            for i in range(len(trainingData)):
                # Forward propagation
                l0 = tData[i]
                l0 = l0.reshape(1, -1)

                l1 = self.sigmoid(np.dot(l0, wl0), False)
                l2 = self.sigmoid(np.dot(l1, wl1), False)
                #   print l0.shape, l1.shape, l2.shape

                # Compute output error value
                #   print "Computing error values..."
                output_error = np.subtract(tLabels[i], l2)
                delta2 = np.multiply(output_error, self.sigmoid(l2, True))
                #   print output_error.shape, delta2.shape
                delta2 = delta2 * learning

                # Backward propagation
                #   print "Backwards propagating..."
                hidden_error = delta2.dot(np.transpose(wl1))
                delta1 = hidden_error * self.sigmoid(l1, True)
                #   print hidden_error.shape, delta1.shape
                delta1 = delta1 * learning

                # Update weights
                #   print "Updating weights..."
                #   print l0.shape, delta1.shape
                #   print l1.shape, delta2.shape
                wl0 += np.transpose(l0).dot(delta1)
                wl1 += np.transpose(l1).dot(delta2)

        self.l0 = wl0
        self.l1 = wl1

    def classify(self, data):
        guesses = []
        l2s = []
        # print self.l1
        # print self.l0
        for datum in data:
            datum = datum.items()
            temp1 = []
            for j in range(len(datum)):
                # Only appending the feature at that point
                # Number of features should be the same for all training samples
                temp1.append(datum[j][1])
            sample = np.asarray(temp1)
            sample = sample.flatten()

            # Predicting
            l0 = sample  # np.asarray([sum(temp1)])
            l1 = self.sigmoid(np.dot(l0, self.l0), False)
            l2 = self.sigmoid(np.dot(l1, self.l1), False)
            # print(l0)
            # print(l1)
            # print(l2)
            # print(np.argmax(l2))
            # exit()
            output = np.argmax(l2)
            l2s.append(l2)
            guesses.append(output)
        return guesses
