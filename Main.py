# Jakob Sunde
# jsunde
# Peter Giseburt
# petergg
# 2/15/2017

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import string, math, operator, sys, numpy

colors = ['pink', 'lightblue', 'thistle']

# Stores the content of the papers
papers = ['']

# Stores a list of the paper numbers that correspond to each author
authorsToPaperNumbers = {'HAMILTON': [], 'JAY': [], 'MADISON': [], 'DISPUTED': []}

# Stores the names of the authors, including 'DISPUTED'
authors = ['HAMILTON', 'MADISON', 'JAY']

# Set disputed papers
for i in range(49, 59):
	authorsToPaperNumbers['DISPUTED'].append(i)

authorsToPaperNumbers['DISPUTED'].append(62)
authorsToPaperNumbers['DISPUTED'].append(63)

# Set joint papers
for i in range(18, 21):
	authorsToPaperNumbers['HAMILTON'].append(i)
	authorsToPaperNumbers['MADISON'].append(i)


def readData(fileName):
	global authorsToPaperNumbers
	f = open(fileName, 'r')

	lines = [line.lstrip().rstrip().translate(None, string.punctuation) for line in f.readlines()]
	start = -1
	currAuthor = ''
	currPaperNum = 0

	for i in range(len(lines)):
		# Remove punctuation from line
		line = lines[i]
		
		# If we're not currently in the middle of a paper
		# check for the start of a paper
		if line.startswith('FEDERALIST'):
			currPaperNum = int(line.split(' ')[-1])

		if start == -1:
			for author in authors:
				if line.startswith(author):
					currAuthor = author					

					# Set start to be four lines later, so as not to include author name
					# or paper header ('To the people of ...') in analysis of paper contents
					start = i + 4
					break

		# If at the end of a paper, add contents of paper to papers
		# and corresponding paper number to list for current author
		if line.startswith('PUBLIUS'):
			# If not disputed or joint,
			# store the current paper number that corresponds to the current author
			if currPaperNum not in authorsToPaperNumbers[currAuthor]:
				authorsToPaperNumbers[currAuthor].append(currPaperNum)
			
			# Add content of current paper to papers
			papers.append(lines[start:i])

			# Set start to -1 to indicate we are not in the middle of a paper
			start = -1

	authorsToPaperNumbers = {k: sorted(v) for k, v in authorsToPaperNumbers.items()}


# Returns a list of tuples containing the top n words and their relative frequency
def topWords(n):
	wordFrequencies = {}

	totalWords = 0

	for paper in papers:
		for line in paper:
			for word in line.split(' '):
				if word == '':
					continue

				word = word.lower()
				totalWords+=1

				# Add 1 to word frequncy, paremeter of 0 is the default value,
				# if that word isn't in the dictionary yet
				wordFrequencies[word] = wordFrequencies.get(word, 0) + 1

	# Map frequencies to be relative to the total number of words
	wordFrequencies = {k: 1.0 * v / totalWords for k, v in wordFrequencies.items()}
	wordFrequencies = sorted(wordFrequencies.items(), key=operator.itemgetter(1), reverse=True)
	wordFrequencies = [x[0] for x in wordFrequencies]
	return wordFrequencies


# Returns a map from papers to top words to their frequences
def tf():
	papersToWordsToFrequencies = {}

	for paperNum in range(1, len(papers)):
		paper = papers[paperNum]
		totalWords = 0
		wordsToFrequencies = {}

		for line in paper:
			for word in line.split(' '):
				word = word.lower()

				totalWords+=1

				
				wordsToFrequencies[word] = wordsToFrequencies.get(word, 0) + 1

		papersToWordsToFrequencies[paperNum] = {k: 1.0 * v / totalWords for k, v in wordsToFrequencies.items()}
	return papersToWordsToFrequencies


# Returns a map from authors to words to frequencies for the top N words that have the greatest disparity of usage between authors
def sampleForTopN(words, papersToWordsToFrequencies, n):
	authorsToWordsToFrequencies = {'HAMILTON': {}, 'JAY': {}, 'MADISON': {}}
	avgWordsToFrequencies = {}

	for paperNum in range(1, len(papers)):
		for word in words:
			wordFreq = papersToWordsToFrequencies[paperNum].get(word, 0)
			avgWordsToFrequencies[word] = avgWordsToFrequencies.get(word, 0) + wordFreq

			for author in authors:
				if paperNum in authorsToPaperNumbers[author]:
					authorsToWordsToFrequencies[author][word] = authorsToWordsToFrequencies[author].get(word, 0) + wordFreq

	authorsToWordsToFrequencies = {k1: {k2: v2 / len(authorsToPaperNumbers[k1]) for k2, v2 in v1.items()} for k1, v1 in authorsToWordsToFrequencies.items()}
	avgWordsToFrequencies = {k: v / (len(papers) - 1) for k, v in avgWordsToFrequencies.items()}

	wordsToDistanceFromAvg = {}
	for word in words:
		wordsToDistanceFromAvg[word] = 0
		for author in authors:
			avg = avgWordsToFrequencies[word]
			curr = authorsToWordsToFrequencies[author][word]
			wordsToDistanceFromAvg[word] += math.fabs(avg - curr)

	minVal = sorted(wordsToDistanceFromAvg.items(), key=operator.itemgetter(1), reverse=True)[n][1]
	bestWords = [k for k, v in wordsToDistanceFromAvg.items() if v > minVal]

	return {author: {word: freq for word, freq in wordsToFreqs.items() if word in bestWords} for author, wordsToFreqs in authorsToWordsToFrequencies.items()}


# Return the error on the training or test dataset based on the isTest flag
def kMeans(authorsToSamples, papersToWordsToFrequencies, isTest):
	paperNums = range(1, len(papers))
	if isTest:
		paperNums = authorsToPaperNumbers['DISPUTED']
		hamilton = 0
		madison = 0
		jay = 0

	error = 0.0
	for paperNum in paperNums:
		predictedAuthor = kMeansPredict(authorsToSamples, papersToWordsToFrequencies[paperNum])

		if isTest:
			if predictedAuthor == 'HAMILTON':
				hamilton += 1
			elif predictedAuthor == 'MADISON':
				madison += 1
			else:
				jay += 1
		elif paperNum not in authorsToPaperNumbers[predictedAuthor]:
			error += 1

	if isTest:
		return (hamilton, madison, jay)
	else:
		return error * 100 / len(paperNums)


def kMeansPredict(authorsToSamples, paperWordsToFreqs):
	hDist = computeDist(authorsToSamples['HAMILTON'], paperWordsToFreqs)
	mDist = computeDist(authorsToSamples['MADISON'], paperWordsToFreqs)
	jDist = computeDist(authorsToSamples['JAY'], paperWordsToFreqs)

	return predict(hDist, mDist, jDist)
	

# Simply assign each disputed paper to the author of the closest paper in the non disputed set
def knn(papersToWordsToFrequencies):
	# TODO: implement
	paperNums = authorsToPaperNumbers['DISPUTED']

	hamilton = 0
	madison = 0
	jay = 0

	for paperNum in paperNums:
		predictedAuthor = knnPredict(papersToWordsToFrequencies, paperNum)

		if predictedAuthor == 'HAMILTON':
			hamilton += 1
		elif predictedAuthor == 'MADISON':
			madison += 1
		else:
			jay += 1

	return (hamilton, madison, jay)


def knnPredict(papersToWordsToFrequencies, num):
	# TODO: implement
	paper = papersToWordsToFrequencies[num]
	hDist = min([computeDist(papersToWordsToFrequencies[paperNum], paper) for paperNum in authorsToPaperNumbers['HAMILTON']])
	mDist = min([computeDist(papersToWordsToFrequencies[paperNum], paper) for paperNum in authorsToPaperNumbers['MADISON']])
	jDist = min([computeDist(papersToWordsToFrequencies[paperNum], paper) for paperNum in authorsToPaperNumbers['JAY']])

	return predict(hDist, mDist, jDist)


# Return the name of the author with the smallest dist
def predict(hDist, mDist, jDist):
	minDist = min(hDist, mDist, jDist)

	if hDist == minDist:
		return 'HAMILTON'
	elif mDist == minDist:
		return 'MADISON'
	else:
		return 'JAY'


# Accepts two dicts, d1 and d2, that map from words to frequencies, and returns the "distance" between them
# d1 should be the sample dict
def computeDist(d1, d2):
	dist = 0.0
	for word in d1:
		dist += (d1[word] - d2.get(word, 0)) ** 2

	return math.sqrt(dist)


def plot(title, xLabel, yLabel, x, y, seriesLabels=None, bar=False, tickLabels=None):
	fig = plt.figure()

	plt.title(title)
	plt.xlabel(xLabel)
	plt.ylabel(yLabel)
	

	if bar:
		width = 0.3	

		for i in range(len(seriesLabels)):
			plt.bar(x + i * width, y[i], width, label=seriesLabels[i], color=colors[i])

		plt.xticks(x + width / 2, tickLabels, rotation=45)
		plt.legend(borderaxespad=1, fontsize=12)
	elif seriesLabels:
		for i in range(len(seriesLabels)):
			plt.plot(x, y[i], label=seriesLabels[i])
	
		plt.legend(borderaxespad=1, fontsize=12)
		plt.ylim(0, 12)
	else:
		plt.ylim(0, 50)

		plt.plot(x, y)

	plt.savefig('%s.png' % title, format='png')
	plt.show()


def main():
	readData('papers.txt')

	# Remove joint papers from both Hamilton and Madison paper lists
	for i in range(18, 21):
		authorsToPaperNumbers['HAMILTON'].remove(i)
		authorsToPaperNumbers['MADISON'].remove(i)

	topNWords = topWords(50)
	papersToWordsToFrequencies = tf()

	Ns = range(1, 30)
	seriesLabels = ['Hamilton', 'Madison', 'Jay']

	# Output top words
	if '-w' in sys.argv:
		authorsToSamples = sampleForTopN(topNWords, papersToWordsToFrequencies, 15)
		tickLabels = authorsToSamples['HAMILTON'].keys()
		data = []
		data.append([v for k, v in authorsToSamples['HAMILTON'].items()])
		data.append([v for k, v in authorsToSamples['MADISON'].items()])
		data.append([v for k, v in authorsToSamples['JAY'].items()])

		plot('Words With the Most Varied Usage Across Authors', 'Words', 'Frequency', numpy.arange(15), data, seriesLabels, True, tickLabels)

	# Output training error
	elif '-t' in sys.argv or '--train' in sys.argv:
		
		# For KMeans
		if 'KMeans' in sys.argv:
			error = []
			for i in Ns:
				authorsToSamples = sampleForTopN(topNWords, papersToWordsToFrequencies, i)
				error.append(kMeans(authorsToSamples, papersToWordsToFrequencies, False))
			
			plot('Training Error for Number of Words (No Joint)', 'Number of Words in Sample', 'Training Error: Incorrect Author Predictions (%)', Ns, error)

	# Run classification
	elif '-r' in sys.argv or '--run' in sys.argv:

		# Using KMeans
		if 'KMeans' in sys.argv:
			predictions = []
			for i in Ns:
				authorsToSamples = sampleForTopN(topNWords, papersToWordsToFrequencies, i)
				predictions.append(kMeans(authorsToSamples, papersToWordsToFrequencies, True))

			predictions = [[x[0] for x in predictions], [x[1] for x in predictions], [x[2] for x in predictions]]
			print predictions
			plot('Predictions for Disputed Papers (No Joint)', 'Number of Words in Sample', 'Number of Papers', Ns, predictions, seriesLabels)

		# Using KNN
		elif 'KNN' in sys.argv:
			predictions = []

			predictions = knn(papersToWordsToFrequencies)

			# TODO: Maybe Plot? right now there's only one output and it's that
			# Madison wrote all the disputed papers

	

if __name__ == '__main__':
    main()