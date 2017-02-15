# Jakob Sunde
# jsunde
# Peter Giseburt
# petergg
# 2/15/2017

papers = {'HAMILTON': [], 'JAY': [], 'MADISON': [], 'UNKNOWN': []}
authors = papers.keys()

def readData(fileName):
	f = open(fileName, 'r')

	lines = f.readlines()
	start = -1
	currAuthor=''
	for i in range(len(lines)):
		line = lines[i].lstrip().rstrip()
		
		# If we're not currently in the middle of a paper
		# check for the start of a paper
		# TODO: add logic to keep track of which federalist paper this is
		if start == -1:
			for author in authors:
				if line.startswith(author):
					currAuthor = author
					start = i
					break

		if line == 'PUBLIUS':
			papers[currAuthor].append(lines[start:i])

			# Set start to -1 to indicate we are not in the middle of a paper
			start = -1

def main():
	readData('papers.txt')
	
if __name__ == '__main__':
    main()