from __future__ import print_function
# dataClassifier.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# This file contains feature extraction methods and harness
# code for data classification
import random
import time

import numpy
import mostFrequent
import naiveBayes
import perceptron
import mlp
import samples
import sys
import util

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 70


def basicFeatureExtractorDigit(datum):
    """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features


def basicFeatureExtractorFace(datum):
    """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features


def enhancedFeatureExtractorDigit(datum):
    """
  Your feature extraction playground.

  You should return a util.Counter() of features
  for this datum (datum is of type samples.Datum).

  ## DESCRIBE YOUR ENHANCED FEATURES HERE...

  ##
  """
    features = basicFeatureExtractorDigit(datum)

    "*** YOUR CODE HERE ***"

    return features


def contestFeatureExtractorDigit(datum):
    """
  Specify features to use for the minicontest
  """
    features = basicFeatureExtractorDigit(datum)
    return features


def enhancedFeatureExtractorFace(datum):
    """
  Your feature extraction playground for faces.
  It is your choice to modify this.
  """
    features = basicFeatureExtractorFace(datum)
    return features


def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
    """
  This function is called after learning.
  Include any code that you want here to help you analyze your results.

  Use the printImage(<list of pixels>) function to visualize features.

  An example of use has been given to you.

  - classifier is the trained classifier
  - guesses is the list of labels predicted by your classifier on the test set
  - testLabels is the list of true labels
  - testData is the list of training datapoints (as util.Counter of features)
  - rawTestData is the list of training datapoints (as samples.Datum)
  - printImage is a method to visualize the features
  (see its use in the odds ratio part in runClassifier method)

  This code won't be evaluated. It is for your own optional use
  (and you can modify the signature if you want).
  """

    # Put any code here...
    # Example of use:
    for i in range(len(guesses)):
        prediction = guesses[i]
        truth = testLabels[i]
        if (prediction != truth):
            print("===================================")
            print("Mistake on example %d" % i)
            print("Predicted %d; truth is %d" % (prediction, truth))
            print("Image: ")
            print(rawTestData[i])
            break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def printImage(self, pixels):
        """
      Prints a Datum object that contains all pixels in the
      provided list of pixels.  This will serve as a helper function
      to the analysis function you write.

      Pixels should take the form
      [(2,2), (2, 3), ...]
      where each tuple represents a pixel.
      """
        image = samples.Datum(None, self.width, self.height)
        for pix in pixels:
            try:
                # This is so that new features that you could define which
                # which are not of the form of (x,y) will not break
                # this image printer...
                x, y = pix
                image.pixels[x][y] = 2
            except:
                print("new features:", pix)
                continue
        print(image)


def default(str):
    return str + ' [Default: %default]'


def readCommand(argv):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'),
                      choices=['mostFrequent', 'nb', 'naiveBayes', 'perceptron', 'mira', 'minicontest', 'mlp'],
                      default='mostFrequent')
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='digits')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
    parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False,
                      action="store_true")
    parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False,
                      action="store_true")
    parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
    parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
    parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
    parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"),
                      type="float", default=1.0)
    parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False,
                      action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print("Doing classification")
    print("--------------------")
    print("data:\t\t" + options.data)
    print("classifier:\t\t" + options.classifier)
    if not options.classifier == 'minicontest':
        print("using enhanced features?:\t" + str(options.features))
    else:
        print("using minicontest feature extractor")
    print("training set size:\t" + str(options.training))
    if (options.data == "digits"):
        printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorDigit
        else:
            featureFunction = basicFeatureExtractorDigit
        if (options.classifier == 'minicontest'):
            featureFunction = contestFeatureExtractorDigit
    elif (options.data == "faces"):
        printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorFace
        else:
            featureFunction = basicFeatureExtractorFace
    else:
        print("Unknown dataset", options.data)
        print(USAGE_STRING)
        sys.exit(2)

    if (options.data == ""
                        "digits"):
        legalLabels = range(10)
    else:
        legalLabels = range(2)

    if options.training <= 0:
        print("Training set size should be a positive integer (you provided: %d)" % options.training)
        print(USAGE_STRING)
        sys.exit(2)

    if options.smoothing <= 0:
        print("Please provide a positive number for smoothing (you provided: %f)" % options.smoothing)
        print(USAGE_STRING)
        sys.exit(2)

    if options.odds:
        if options.label1 not in legalLabels or options.label2 not in legalLabels:
            print("Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2))
            print(USAGE_STRING)
            sys.exit(2)

    if (options.classifier == "mostFrequent"):
        classifier = mostFrequent.MostFrequentClassifier(legalLabels)
    elif (options.classifier == "naiveBayes" or options.classifier == "nb"):
        classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
        classifier.setSmoothing(options.smoothing)
        if (options.autotune):
            print("using automatic tuning for naivebayes")
            classifier.automaticTuning = True
        else:
            print("using smoothing parameter k=%f for naivebayes" % options.smoothing)
    elif (options.classifier == "perceptron"):
        classifier = perceptron.PerceptronClassifier(legalLabels, options.iterations)
    elif (options.classifier == "mlp"):
        classifier = mlp.MLPClassifier(legalLabels, options.iterations)
    elif (options.classifier == 'minicontest'):
        import minicontest
        classifier = minicontest.contestClassifier(legalLabels)
    else:
        print("Unknown classifier:", options.classifier)
        print(USAGE_STRING)

        sys.exit(2)

    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    args['printImage'] = printImage

    return args, options


USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """


# Main harness code

def runClassifier(args, options):
    sample = open('printResults.txt', 'w')
    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']

    # Load data
    if (options.data == "faces"):
        numTest = 150
        numTraining = 451
        rawTrainingData = samples.loadDataFile("data/facedata/facedatatrain", numTraining, FACE_DATUM_WIDTH,
                                               FACE_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("data/facedata/facedatatrainlabels", numTraining)
        rawValidationData = samples.loadDataFile("data/facedata/facedatatrain", numTest, FACE_DATUM_WIDTH,
                                                 FACE_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("data/facedata/facedatatrainlabels", numTest)
        rawTestData = samples.loadDataFile("data/facedata/facedatatest", numTest, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("data/facedata/facedatatestlabels", numTest)
    else:
        numTest = 1000
        numTraining = 5000
        rawTrainingData = samples.loadDataFile("data/digitdata/trainingimages", numTraining, DIGIT_DATUM_WIDTH,
                                               DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("data/digitdata/traininglabels", numTraining)
        rawValidationData = samples.loadDataFile("data/digitdata/validationimages", numTest, DIGIT_DATUM_WIDTH,
                                                 DIGIT_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("data/digitdata/validationlabels", numTest)
        rawTestData = samples.loadDataFile("data/digitdata/testimages", numTest, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("data/digitdata/testlabels", numTest)

    # Extract features
    print("Extracting features...")
    trainingData = map(featureFunction, rawTrainingData)
    validationData = map(featureFunction, rawValidationData)
    testData = map(featureFunction, rawTestData)
    for percent in range(1, 11):

        time_total = 0
        total_correct_val = 0
        total_correct_test = 0
        values_test = [0 for x in range(5)]

        for y in range(5):

            randomIndices = random.sample(range(numTraining), int((percent * .1) * numTraining))
            randomTrainingData = numpy.take(trainingData, randomIndices)
            randomTrainingLabels = numpy.take(trainingLabels, randomIndices)

            # Conduct training and testing
            if not (options.autotune and options.classifier == "naiveBayes"):
                start = time.time()

                classifier.train(randomTrainingData, randomTrainingLabels, validationData, validationLabels)
                end = time.time()
                reqtime = end - start
                time_total += reqtime

                guesses = classifier.classify(validationData)
                correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
                total_correct_val += correct


            elif options.autotune:
                kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
                bestK = 0
                bestCorrect = 0
                for k in kgrid:
                    classifier.k = k
                    print("Training Using " + str(percent * 10) + "% of The Data k =" + str(k), file=sample)
                    classifier.train(randomTrainingData, randomTrainingLabels, validationData, validationLabels)
                    print("Validating Using " + str(percent * 10) + "% of The Data k =" + str(k), file=sample)
                    guesses = classifier.classify(validationData)
                    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
                    if correct > bestCorrect:
                        bestCorrect = correct
                        bestK = k
                    print(str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (
                                100.0 * correct / len(validationLabels)), file=sample)
                classifier.k = bestK
                print("Training Using " + str(percent * 10) + "% of The Data Best K: " + str(bestK), file=sample)
                classifier.train(randomTrainingData, randomTrainingLabels, validationData, validationLabels)
                print("Validating Using " + str(percent * 10) + "% of The Data Best K: " + str(bestK), file=sample)
                guesses = classifier.classify(validationData)
                correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
                print(str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (
                            100.0 * correct / len(validationLabels)), file=sample)

            guesses = classifier.classify(testData)
            correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
            values_test[y] = correct
            print(values_test)
            total_correct_test += correct

        print("Training Using " + str(percent * 10) + "% of The Data", file=sample)
        print("Time : " + str(time_total / 5), file=sample)

        print("Validating Using " + str(percent * 10) + "% of The Data aka " + str(len(randomIndices)) + " images",
              file=sample)
        print(str(total_correct_val / 5), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (
                100.0 * correct / len(validationLabels)), file=sample)

        print("Testing Using " + str(percent * 10) + "% of The Data", file=sample)

        mean = total_correct_test / 5
        print(100.0 * correct / len(testLabels))
        print(str(mean),
              ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)),
              file=sample)

        # standard deviation
        variance = sum([((x - mean) ** 2) for x in values_test]) / 5
        res = variance ** 0.5

        print("standard deviation of", res, "\n\n", file=sample)

        # analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

        # do odds ratio computation if specified at command line
        if ((options.odds) & (options.classifier == "naiveBayes" or (options.classifier == "nb"))):
            label1, label2 = options.label1, options.label2
            features_odds = classifier.findHighOddsFeatures(label1, label2)
            if (options.classifier == "naiveBayes" or options.classifier == "nb"):
                string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
            else:
                string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)

            print(string3)
            printImage(features_odds)

        if ((options.weights) & (options.classifier == "perceptron")):
            for l in classifier.legalLabels:
                features_weights = classifier.findHighWeightFeatures(l)
                print("=== Features with high weight for label %d ===" % l, file=sample)
                printImage(features_weights)


if __name__ == '__main__':
    # Read input
    args, options = readCommand(sys.argv[1:])
    # Run classifier
    runClassifier(args, options)
