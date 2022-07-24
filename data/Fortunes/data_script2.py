import requests
from bs4 import BeautifulSoup



outFile = "data/Fortunes/data.txt"
siteName = "http://www.fortunecookiemessage.com/archive.php?start="
low = 0
high = 839
skip = 50



# Open the file
file = open(outFile, "a", encoding="utf-8")

# Get all fortunes
for cur in range(low, high, skip):
    # Load a new page
    r = requests.get(siteName + str(cur))
    soup = BeautifulSoup(r.text, 'lxml')

    # Get all tr tags and remove the first one
    elements = soup.find_all("tr")[1:]
    
    # Iterate over each element to ge the data
    for el in elements:
        # Get the data
        text = el.contents[1].contents[0].contents[0]
        
        # Save the text to the dataset
        try:
            file.write(text+"\n")
        except:
            continue


# Close the file
file.close()