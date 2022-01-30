import csv
filename = "/home/mbarnig/Downloads/lb-lb/LOD/lexicon-LOD-revised-pipe.txt"
outputname = "/home/mbarnig/Downloads/lb-lb/LOD/lexicon-LOD-espeak-pipe-temp.txt"
newfile = open(outputname, 'w')
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter = '|')
    for row in csvreader:
        #print(row[1])
        replacements = []
        for phoneme in row[1].split():
            #print(phoneme) 
            # translate monophtongs
            if phoneme == "aː" or phoneme == "e" or phoneme == "eː" or phoneme == "i" or phoneme == "iː" or phoneme == "o" or phoneme == "oː" or phoneme == "u" or phoneme == "uː" or phoneme == "y" or phoneme == "yː":
                replacements.append(phoneme + " ")
            elif phoneme == "ɑ":
                replacements.append("a ")
            elif phoneme == "ɛː":
                replacements.append("aE ") 
            elif phoneme == "æ":
                replacements.append("E ")                      
            elif phoneme == "ə":            
                replacements.append("@E ")
            elif phoneme == "ɐ":
                replacements.append("rR ")   
            elif phoneme == "ɑ̃ː":
                replacements.append("eA ")  
            elif phoneme == "ɛ̃ː":
                replacements.append("iA ")
            elif phoneme == "õː":            
                replacements.append("oA ")
            elif phoneme == "œː":
                replacements.append("OU ")      
            # translate diphtongs    
            elif phoneme == "æːɪ":
                replacements.append("aI ")
            elif phoneme == "ɑʊ":
                replacements.append("aU ") 
            elif phoneme == "æːʊ":
                replacements.append("AU ")                      
            elif phoneme == "ɑɪ":            
                replacements.append("eI ")
            elif phoneme == "ɜɪ":
                replacements.append("OI ")   
            elif phoneme == "oɪ":
                replacements.append("eU ")  
            elif phoneme == "iə":
                replacements.append("iE ")
            elif phoneme == "əʊ":            
                replacements.append("oU ")
            elif phoneme == "uə":
                replacements.append("uE ") 
            # translate and replace consonants
            # nasals                                      
            elif phoneme == "m" or phoneme == "n":
                replacements.append(phoneme + " ") 
            elif phoneme == "ŋ":
                replacements.append("N ")  
            # plosives         
            elif phoneme == "b" or phoneme =="p" or phoneme == "d" or phoneme == "t" or phoneme == "g" or phoneme == "k":
                replacements.append(phoneme + " ")    
            # affricates
            elif phoneme == "ts":
                replacements.append("TS ")            
            elif phoneme == "dʒ":
                replacements.append("dZ ")   
            # fricatives               
            elif phoneme == "f" or phoneme == "v" or phoneme == "w" or phoneme == "s" or phoneme == "z" or phoneme == "h":   
                replacements.append(phoneme)
            elif phoneme == "ʃ":
                replacements.append("S ")                 
            elif phoneme == "ʒ":
                replacements.append("J ")                 
            elif phoneme == "χ":
                replacements.append("x ")                 
            elif phoneme == "ɕ":
                replacements.append("X ")                   
            elif phoneme == "ʁ":
                replacements.append("rR ")                 
            elif phoneme == "ʑ":
                replacements.append("Z ")      
            # Approximants           
            elif phoneme == "l" or phoneme == "j":
                replacements.append(phoneme + " ")
            # Trill
            elif phoneme == "ʀ":
                replacements.append("r ")   
            # convert allophones    
            elif phoneme == "øː":
                replacements.append("OU ")
            elif phoneme == "ɔː":
                replacements.append("oː ") 
            elif phoneme == "ɔ":
                replacements.append("o ")                      
            elif phoneme == "ɛ":            
                replacements.append("e ")
            elif phoneme == "œ":
                replacements.append("OU ")  
            elif phoneme == "ɳ":
                replacements.append("N ")  
            elif phoneme == "ɲ":    
                replacements.append("N ")                       
            else:
                print("*** phoneme to replace not found ***" + phoneme) 
        newrow = "".join(replacements)
        # print(newrow)                           
        newfile.write(row[0] + "|" + newrow + "\n")
csvfile.close()
newfile.close()
