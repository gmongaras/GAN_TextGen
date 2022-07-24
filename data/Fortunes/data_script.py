
import csv


in_file = "data/Fortunes/fortunes.txt"
out_file = "data/Fortunes/data.txt"



with open(in_file, "r", encoding="utf-8") as f:
    out = open(out_file, "w", encoding="utf-8")
    
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
    out.close()
