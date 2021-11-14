import csv
filename = "/home/mbarnig/Downloads/lb-lb/LOD/lexicon-LOD-espeak-pipe-temp.txt"
outputname = "/home/mbarnig/Downloads/lb-lb/LOD/lexicon-LOD-espeak-pipe.txt"
newfile = open(outputname, 'w')
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter = '|')
    for row in csvreader:
        replacements = []
        for phoneme in row[1]:
            if phoneme == "ː":
                replacements.append(":")
            else:
                replacements.append(phoneme)
        newrow = "".join(replacements)
        # print(newrow)                           
        newfile.write(row[0] + "|" + newrow + "\n")
csvfile.close()
newfile.close()
