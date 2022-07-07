
import csv


in_file = "data/fortunes"
out_file = "data.txt"



with open(in_file, "r") as f:
    out = open(out_file, "w")
    
    reader = csv.reader(f, delimiter='%')
    for row in reader:
        if len(row) < 1:
            continue
        r = row[0].strip()
        if len(r) < 3:
            continue
        if (r[0] == "-" and r[1] == "-"):
            continue
        out.write(r + "\n")
