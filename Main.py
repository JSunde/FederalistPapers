# Jakob Sunde
# jsunde
# Peter Giseburt
# petergg
# 2/15/2017

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import string, math, operator, sys, numpy

colors = ['pink', 'lightblue', 'thistle', 'lightgreen', 'paleturquoise']
markers = ['o', 's', 'D', '^']
linestyles = ['-', '--', ':', '-.']

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

jayPresent = False

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


# Returns a list of tuples containing the words and their relative frequency
def wordsFreqs():
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
# Accepts a list of paper numbers, for which to perform the 
def tf(paperNums):
	papersToWordsToFrequencies = {}

	for paperNum in paperNums:
		paper = papers[paperNum]
		totalWords = 0
		wordsToFrequencies = {}

		for line in paper:
			for word in line.split(' '):
				word = word.lower()

				if word != '':
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
					if jayPresent or (not jayPresent and author != 'JAY'):
						authorsToWordsToFrequencies[author][word] = authorsToWordsToFrequencies[author].get(word, 0) + wordFreq


	authorsToWordsToFrequencies = {k1: {k2: v2 / len(authorsToPaperNumbers[k1]) for k2, v2 in v1.items()} for k1, v1 in authorsToWordsToFrequencies.items()}
	avgWordsToFrequencies = {k: v / (len(papers) - 1) for k, v in avgWordsToFrequencies.items()}

	wordsToDistanceFromAvg = {}
	for word in words:
		wordsToDistanceFromAvg[word] = 0
		for author in authors:
			if jayPresent or (not jayPresent and author != 'JAY'):
				avg = avgWordsToFrequencies[word]
				curr = authorsToWordsToFrequencies[author][word]
				wordsToDistanceFromAvg[word] += math.fabs(avg - curr)

	minVal = sorted(wordsToDistanceFromAvg.items(), key=operator.itemgetter(1), reverse=True)[n][1]
	bestWords = [k for k, v in wordsToDistanceFromAvg.items() if v > minVal]

	result = {author: {word: freq for word, freq in wordsToFreqs.items() if word in bestWords} for author, wordsToFreqs in authorsToWordsToFrequencies.items()}

	sortedResult = {}
	for author in result.keys():
		wordsToFreqs = result[author]
		wordsToFreqs = {word: (freq, wordsToDistanceFromAvg[word]) for word, freq in wordsToFreqs.items()}
		sortedResult[author] = sorted(wordsToFreqs.items(), key=lambda x: x[1][1], reverse=True)

	return (result, bestWords, sortedResult)


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
		elif paperNum not in authorsToPaperNumbers[predictedAuthor] and paperNum not in authorsToPaperNumbers['DISPUTED'] and paperNum not in [18, 19, 20]:
			if jayPresent or paperNum not in authorsToPaperNumbers['JAY']:
				error += 1


	if isTest:
		return (hamilton, madison, jay)
	else:
		if jayPresent:
			return error * 100 / (len(paperNums) - len(authorsToPaperNumbers['DISPUTED']))
		else:
			return error * 100 / (len(paperNums) - len(authorsToPaperNumbers['DISPUTED']) - len(authorsToPaperNumbers['JAY']))


def kMeansPredict(authorsToSamples, paperWordsToFreqs):
	hDist = computeDist(authorsToSamples['HAMILTON'], paperWordsToFreqs)
	mDist = computeDist(authorsToSamples['MADISON'], paperWordsToFreqs)
	
	if jayPresent:
		jDist = computeDist(authorsToSamples['JAY'], paperWordsToFreqs)
		return predict(hDist, mDist, jDist)

	return predict(hDist, mDist, sys.maxint)


# Simply assign each disputed paper to the author of the closest paper in the non disputed set
def KNN(papersToWordsToFrequencies, k, isTest, isValidation=False):
	paperNums = range(1, len(papers))
	if isTest:
		paperNums = authorsToPaperNumbers['DISPUTED']
		hamilton = 0
		madison = 0
		jay = 0

	hamilton = 0
	madison = 0
	jay = 0

	error = 0.0
	if isValidation:
		hLen = len(authorsToPaperNumbers['HAMILTON'])
		mLen = len(authorsToPaperNumbers['MADISON'])
		jLen = len(authorsToPaperNumbers['JAY'])

		paperNums = []
		paperNums += authorsToPaperNumbers['HAMILTON'][-(hLen / 4):]
		paperNums += authorsToPaperNumbers['MADISON'][-(mLen / 4):]

	for paperNum in paperNums:
		predictedAuthor = KNNPredict(papersToWordsToFrequencies, paperNum, k, isValidation)
		if isTest and not isValidation:
			if predictedAuthor == 'HAMILTON':
				hamilton += 1
			elif predictedAuthor == 'MADISON':
				madison += 1
			else:
				jay += 1
		elif paperNum not in authorsToPaperNumbers[predictedAuthor] and paperNum not in authorsToPaperNumbers['DISPUTED'] and paperNum not in [18, 19, 20]:
			if jayPresent or paperNum not in authorsToPaperNumbers['JAY']:
				print '%s %d' % (predictedAuthor, paperNum)
				error += 1

	if isValidation:
		return error * 100 / (len(paperNums))
	if isTest:
		return (hamilton, madison, jay)
	else:
		if jayPresent:
			return error * 100 / (len(paperNums) - len(authorsToPaperNumbers['DISPUTED']))
		else:
			return error * 100 / (len(paperNums) - len(authorsToPaperNumbers['DISPUTED']) - len(authorsToPaperNumbers['JAY']))


def KNNPredict(papersToWordsToFrequencies, num, k, isValidation=False):
	paper = papersToWordsToFrequencies[num]
	# hDist = min([computeDist(papersToWordsToFrequencies[paperNum], paper) for paperNum in authorsToPaperNumbers['HAMILTON']])
	# mDist = min([computeDist(papersToWordsToFrequencies[paperNum], paper) for paperNum in authorsToPaperNumbers['MADISON']])
	# jDist = min([computeDist(papersToWordsToFrequencies[paperNum], paper) for paperNum in authorsToPaperNumbers['JAY']])
	
	hLen = len(authorsToPaperNumbers['HAMILTON'])
	mLen = len(authorsToPaperNumbers['MADISON'])
	jLen = len(authorsToPaperNumbers['JAY'])

	if isValidation:
		hDists = sorted([computeDist(papersToWordsToFrequencies[paperNum], paper) for paperNum in authorsToPaperNumbers['HAMILTON'][:-(hLen / 4)]])
		mDists = sorted([computeDist(papersToWordsToFrequencies[paperNum], paper) for paperNum in authorsToPaperNumbers['MADISON'][:-(mLen / 4)]])
		
		if jayPresent:
			jDists = sorted([computeDist(papersToWordsToFrequencies[paperNum], paper) for paperNum in authorsToPaperNumbers['JAY'][:-(jLen / 4)]])
	else:
		hDists = sorted([computeDist(papersToWordsToFrequencies[paperNum], paper) for paperNum in authorsToPaperNumbers['HAMILTON']])
		mDists = sorted([computeDist(papersToWordsToFrequencies[paperNum], paper) for paperNum in authorsToPaperNumbers['MADISON']])
		
		if jayPresent:
			jDists = sorted([computeDist(papersToWordsToFrequencies[paperNum], paper) for paperNum in authorsToPaperNumbers['JAY']])

	hVotes = 0
	mVotes = 0
	jVotes = 0

	for i in range(k):
		if jayPresent:
			minDist = min(hDists[0], mDists[0], jDists[0])
		else:
			minDist = min(hDists[0], mDists[0])

		if hDists[0] == minDist:
			hDists.pop(0)
			hVotes += 1
		elif mDists[0] == minDist:
			mDists.pop(0)
			mVotes += 1
		else:
			jDists.pop(0)
			jVotes += 1

	return predict(0 - hVotes, 0 - mVotes, 0 - jVotes)


def flattenDictOfDicts(outerDct, keys):
	result = {}
	
	for paperNum, innerDct in outerDct.items():

		for word, freq in innerDct.items():
			result[word] = result.get(word, 0) + freq

	result = {k : v / len(outerDct) for k, v in result.items() if k in keys}
	return result


def NB(words, isTest):
	totalPapers = len(papers) - 1
	priorH = float(len(authorsToPaperNumbers['HAMILTON'])) / totalPapers
	priorM = float(len(authorsToPaperNumbers['HAMILTON'])) / totalPapers
	priorJ = float(len(authorsToPaperNumbers['HAMILTON'])) / totalPapers

	hWordFreqs = flattenDictOfDicts(tf(authorsToPaperNumbers['HAMILTON']), words)
	mWordFreqs = flattenDictOfDicts(tf(authorsToPaperNumbers['MADISON']), words)
	jWordFreqs = flattenDictOfDicts(tf(authorsToPaperNumbers['JAY']), words)

	hMin = min(hWordFreqs.itervalues()) / 1.25
	mMin = min(hWordFreqs.itervalues()) / 1.25
	jMin = min(hWordFreqs.itervalues()) / 1.25
	
	paperNums = range(1, len(papers))
	if isTest:
		paperNums = authorsToPaperNumbers['DISPUTED']

	hamilton = 0
	madison = 0
	jay = 0
	error = 0.0

	for paperNum in paperNums:
		paper = papers[paperNum]
		probs = [0, 0, 0]

		for line in paper:

			for word in line.split(' '):
				word = word.lower()

				probs[0] += math.log(hWordFreqs.get(word, hMin) * priorH)
				probs[1] += math.log(mWordFreqs.get(word, mMin) * priorM)
				probs[2] += math.log(jWordFreqs.get(word, jMin) * priorM)

		indexOfMax = probs.index(max(probs))

		predictedAuthor = ''
		if indexOfMax == 0:
			hamilton += 1
			predictedAuthor = 'HAMILTON'
		elif indexOfMax == 1:
			predictedAuthor = 'MADISON'
			madison += 1
		else:
			predictedAuthor = 'JAY'
			jay += 1

		if not isTest and paperNum not in authorsToPaperNumbers[predictedAuthor] and paperNum not in authorsToPaperNumbers['DISPUTED'] and paperNum not in [18, 19, 20]:
			if jayPresent or paperNum not in authorsToPaperNumbers['JAY']:
				error += 1

	if isTest:
		return (hamilton, madison, jay)
	else:
		if jayPresent:
			return error * 100 / (len(paperNums) - len(authorsToPaperNumbers['DISPUTED']))
		else:
			return error * 100 / (len(paperNums) - len(authorsToPaperNumbers['DISPUTED']) - len(authorsToPaperNumbers['JAY']))


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


def plot(title, xLabel, yLabel, x, y, seriesLabels=None, bar=False, tickLabels=None, alg='', isTest=False, isValidation=False):
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
	elif isTest:
		patches = []
		for i in range(len(seriesLabels)):
			plt.plot(x, y[i], label=seriesLabels[i], color=colors[i])
			patches.append(mpatches.Patch(color=colors[i], label=seriesLabels[i]))
	
		plt.legend(borderaxespad=1, loc=5, fontsize=12, handles=patches)
		plt.ylim(0, 12.5)
	elif isValidation:
		patches = []
		for i in range(len(y)):
			patches.append(mpatches.Patch(color=colors[i % len(colors)], label=seriesLabels[i]))

		plt.legend(borderaxespad=1, handles=patches)

		plt.ylim(0, 15)
		for i in range(len(y)):
			plt.plot(x, y[i], color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])
	else:
		labels = ['KMeans', 'KNN', 'Naive Bayes']

		patches = []
		for i in range(len(y)):
			patches.append(mpatches.Patch(color=colors[i % len(y)], label=labels[i]))

		plt.legend(borderaxespad=1, handles=patches)

		plt.ylim(0, 25)
		for i in range(len(y)):
			plt.plot(x, y[i], color=colors[i % len(y)], linestyle=linestyles[i % len(y)])

	plt.savefig('%s.png' % title, format='png')
	plt.show()


def main():
	readData('papers.txt')

	sys.argv = [word.lower() for word in sys.argv]

	n = 50
	if '-n' in sys.argv:
		n = int(sys.argv[sys.argv.index('-n') + 1])

	if '-j' in sys.argv:
		global jayPresent
		jayPresent = True

	kForKNN = 1
	if '-k' in sys.argv:
		kForKNN = int(sys.argv[sys.argv.index('-k') + 1])

	# Remove joint papers from both Hamilton and Madison paper lists
	for i in range(18, 21):
		authorsToPaperNumbers['HAMILTON'].remove(i)
		authorsToPaperNumbers['MADISON'].remove(i)

	words = wordsFreqs()
	papersToWordsToFrequencies = tf(range(1, len(papers)))

	Ns = range(1, n)
	seriesLabels = ['Hamilton', 'Madison', 'Jay']


	# Output top words
	if '-w' in sys.argv:
		authorsToSamples, bestWords, sortedAuthorsToSamples = sampleForTopN(words, papersToWordsToFrequencies, n)

		print bestWords
		tickLabels = [k for k, v in sortedAuthorsToSamples['HAMILTON']]

		data = []
		print sortedAuthorsToSamples['HAMILTON']
		data.append([v[0] for k, v in sortedAuthorsToSamples['HAMILTON']])
		data.append([v[0] for k, v in sortedAuthorsToSamples['MADISON']])

		if jayPresent:
			data.append([v[0] for k, v in sortedAuthorsToSamples['JAY']])

			plot('Words With the Most Varied Usage Across Authors', 'Words', 'Frequency', numpy.arange(n), data, seriesLabels, True, tickLabels)
		else:
			plot('Words With the Most Varied Usage Across Authors', 'Words', 'Frequency', numpy.arange(n), data, seriesLabels[:-1], True, tickLabels)

	# Output training error
	elif '-t' in sys.argv or '--train' in sys.argv:	
		error = [[], [], []]
		isTest = False
		for i in Ns:
			authorsToSamples, bestWords, sortedAuthorsToSamples = sampleForTopN(words, papersToWordsToFrequencies, i)
			error[0].append(kMeans(authorsToSamples, papersToWordsToFrequencies, isTest))
			knnPapersToWordsToFrequencies = {author: {word: freq for word, freq in wordsToFreqs.items() if word in bestWords}
			 		for author, wordsToFreqs in papersToWordsToFrequencies.items()}
			error[1].append(KNN(knnPapersToWordsToFrequencies, kForKNN, isTest))
			error[2].append(NB(bestWords, isTest))
		
		plot('Training Error as a Function of Number of Feature Words', 'Number of Feature Words', 'Training Error: Incorrect Author Predictions (%)', Ns, error)

	elif '-v' in sys.argv:
		Ks = range(1, int(sys.argv[sys.argv.index('-v') + 1]), 2)

		error = [[] for i in Ks]

		isTest = False
		isValidation = True
		for kIndex in range(len(Ks)):
			print 'k=%d' % Ks[kIndex]
			for i in Ns:
				print '\ti=%d' % i
				authorsToSamples, bestWords, sortedAuthorsToSamples = sampleForTopN(words, papersToWordsToFrequencies, i)

				knnPapersToWordsToFrequencies = {author: {word: freq for word, freq in wordsToFreqs.items() if word in bestWords}
							 		for author, wordsToFreqs in papersToWordsToFrequencies.items()}
				error[kIndex].append(KNN(knnPapersToWordsToFrequencies, Ks[kIndex], isTest, isValidation))

		plot('Validation Error as a Function of Number of Feature Words', 'Number of Feature Words', 'Validation Error: Incorrect Author Predictions (%)', Ns, error, seriesLabels=Ks, isValidation=True)


	# Run classification
	elif '-r' in sys.argv or '--run' in sys.argv:
		predictions = []
		isTest = True
		alg = ''

		# Using KMeans
		predictions = []
		if 'kmeans' in sys.argv:
			alg = 'Kmeans'
			for i in Ns:
				authorsToSamples = sampleForTopN(words, papersToWordsToFrequencies, i)[0]
				predictions.append(kMeans(authorsToSamples, papersToWordsToFrequencies, True))

		# Using KNN
		elif 'knn' in sys.argv:
			alg = 'KNN'
			for i in Ns:
				print i
				bestWords = sampleForTopN(words, papersToWordsToFrequencies, i)[1]
				#{author: {word: freq for word, freq in wordsToFreqs.items() if word in bestWords} for
				 #		author, wordsToFreqs in authorsToWordsToFrequencies.items()}, bestWords)
				papersToWordsToFrequencies = {author: {word: freq for word, freq in wordsToFreqs.items() if word in bestWords}
											  		for author, wordsToFreqs in papersToWordsToFrequencies.items()}
				predictions.append(KNN(papersToWordsToFrequencies, kForKNN, isTest))

		# Using Naive Bayes Net
		elif 'nb' in sys.argv:
			alg = 'Naive Bayes'
			for i in Ns:
				print i
				bestWords = sampleForTopN(words, papersToWordsToFrequencies, i)[1]
				predictions.append(NB(bestWords, isTest))
			
		if jayPresent:
			predictions = [[x[0] for x in predictions], [x[1] for x in predictions], [x[2] for x in predictions]]
			plot('Predictions for Disputed Papers', 'Number of Feature Words', 'Number of Papers', Ns, predictions, seriesLabels=seriesLabels, isTest=isTest)
		else:
			predictions = [[x[0] for x in predictions], [x[1] for x in predictions]]
			plot('Predictions for Disputed Papers using %s' % alg, 'Number of Feature Words', 'Number of Papers', Ns, predictions, seriesLabels=seriesLabels[:-1], isTest=isTest, alg=alg)	

if __name__ == '__main__':
    main()