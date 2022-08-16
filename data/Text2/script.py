import os
from src.helpers.helpers import get_clean_words



indir = "data/Text2/train_data" # Directory with txt files to read in
outFile = "data/Text2/data.txt" # File to put all output data to
maxSentSize = 64     # Max number of words in a sentence to load in
limit = 30000 # Limit on the number of sentences to load in




# The max sentence size should be 2 less than the desired to
# make room for the <START> and <END> tokens
maxSentSize -= 2



# Open the output file
out = open(outFile, "w", encoding="utf-8")


# Open each file in the directory
l = 0
for root, dirs, files in os.walk(indir, topdown=False):
    # Iterate over each file in the directory
    for file in files:
        # Get the filename
        filename = os.path.join(root, file)

        # Open the file
        with open(filename, "r", encoding="utf-8") as f:
            # Read each line from the file
            for line in f:
                # Limit checking
                if l == limit:
                    break

                # Clean the sentence
                sent = get_clean_words(line)
                
                # If the sentence length is less than the max length,
                # save it to the output file
                if len(sent) < maxSentSize:
                    out.write(" ".join(sent) + "\n")
                    l += 1

# Close the output file
out.close()