# Jakob Sunde
# jsunde
# Peter Giseburt
# petergg
# 2/15/2017

import string, math

# Stores the content of the papers
papers = []

# Stores a list of the paper numbers that correspond to each author
authorsToPaperNumbers = {'HAMILTON': [], 'JAY': [], 'MADISON': [], 'DISPUTED': []}

# Stores the names of the authors, including 'DISPUTED'
authors = authorsToPaperNumbers.keys()

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
	f = open(fileName, 'r')

	lines = [line.lstrip().rstrip() for line in f.readlines()]
	start = -1
	currAuthor = ''
	currPaperNum = 0

	for i in range(len(lines)):
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
			for author in authors:
				if currPaperNum not in authorsToPaperNumbers[author]:
					authorsToPaperNumbers[currAuthor].append(currPaperNum)
			
			# Add content of current paper to papers
			papers.append(lines[start:i])

			# Set start to -1 to indicate we are not in the middle of a paper
			start = -1

# Returns a map from words to their tf-idf weight
def tfidf(author):
	wordFrequencies = {}
	idf = {}

	totalWords = 0

	for paperNum in authorsToPaperNumbers[author]:
		paper = papers[paperNum]
		for line in paper:
			# Remove punctuation from line
			line = line.translate(None, string.punctuation)

			for word in line.split(' '):
				totalWords+=1

				# Add 1 to word frequncy, paremeter of 0 is the default value,
				# if that word isn't in the dictionary yet
				wordFrequencies.get(word, 0) + 1

				idf.get(word, set()).add(paperNum)

	# Map frequencies to be relative to the total number of words
	wordFrequencies = {k: v / totalWords for k, v in wordFrequencies.items()}

	# Map idf to be log(N/d) where N is the number of papers this author wrote,
	# and d is the number of papers in which the current word was present
	idf = {k: math.log(len(v) / len(authorsToPaperNumbers[author])) for k, v in idf.items()}

	return {k: v * idf[k] for k, v in wordFrequencies.items() if k in idf.items()}

def main():
	readData('papers.txt')


	authorsToData = {'HAMILTON' : tfidf('HAMILTON'), 'MADISON' : tfidf('MADISON'), 'JAY': tfidf('JAY')}
	
if __name__ == '__main__':
    main()