import csv
filename = "/home/mbarnig/Downloads/lb-lb/LOD/lexicon-LOD-revised-pipe.txt"
outputname = "/home/mbarnig/Downloads/lb-lb/LOD/lexicon-LOD-gruut-pipe.txt"
newfile = open(outputname, 'w')
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter = '|')
    for row in csvreader:
        #print(row[1])
        replacements = []
        for phoneme in row[1].split():
            # translate allophones
            if phoneme == "øː":
                replacements.append("œː ")
            elif phoneme == "ɔː":
                replacements.append("oː ") 
            elif phoneme == "ɔ":
                replacements.append("o ")                      
            elif phoneme == "ɛ":            
                replacements.append("e ")
            elif phoneme == "œ":
                replacements.append("œː ")  
            elif phoneme == "ɳ":
                replacements.append("ŋ ")  
            elif phoneme == "ɲ":
                replacements.append("ŋ ")
            elif phoneme == "ts":
                replacements.append("ʦ ")            
            # keep all other phonemes    
            elif phoneme == "ɑ" or phoneme =="aː" or phoneme == "ɛː" or phoneme == "e" or phoneme == "eː" or phoneme == "æ" or phoneme == "ə" or phoneme == "ɐ":
                replacements.append(phoneme  + " ")
            elif phoneme == "i" or phoneme =="iː" or phoneme == "o" or phoneme == "oː" or phoneme == "u" or phoneme == "uː":
                replacements.append(phoneme + " ")
            elif phoneme == "y" or phoneme =="yː" or phoneme == "ɑ̃ː" or phoneme == "e" or phoneme == "ɛ̃ː" or phoneme == "õː" or phoneme == "œː":
                replacements.append(phoneme + " ")
            elif phoneme == "æːɪ" or phoneme =="ɑʊ" or phoneme == "æːʊ" or phoneme == "ɑɪ" or phoneme == "ɜɪ" or phoneme == "oɪ" or phoneme == "iə" or phoneme == "əʊ" or phoneme == "uə":
                replacements.append(phoneme  + " ")
            #nasals
            elif phoneme == "m" or phoneme =="n" or phoneme == "ŋ":
                replacements.append(phoneme  + " ")
            # plosives         
            elif phoneme == "b" or phoneme =="p" or phoneme == "d" or phoneme == "t" or phoneme == "g" or phoneme == "k":
                replacements.append(phoneme  + " ")
            # affricates    
            elif phoneme == "dʒ":
                replacements.append(phoneme  + " ")   
            # fricatives               
            elif phoneme == "f" or phoneme == "v" or phoneme == "w" or phoneme == "s" or phoneme == "z" or phoneme == "h":   
                replacements.append(phoneme  + " ")
            elif phoneme == "ʃ" or phoneme == "ʒ" or phoneme == "χ" or phoneme == "ɕ" or phoneme == "ʁ" or phoneme == "ʑ":   
                replacements.append(phoneme  + " ")        
            # Approximants           
            elif phoneme == "l" or phoneme == "j":
                replacements.append(phoneme  + " ")
            # Trill
            elif phoneme == "ʀ":
                replacements.append(phoneme  + " ")                         
            else:
                print("*** phoneme to replace not found ***" + phoneme) 
        newrow = "".join(replacements)
        # print(newrow)                           
        newfile.write(row[0] + "|" + newrow + "\n")
csvfile.close()
newfile.close()
