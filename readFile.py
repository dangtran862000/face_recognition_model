# importing csv module
import csv

def find_people(name):
    # csv file name
    filename = "demofile.txt"

    # initializing the titles and rows list
    fields = []
    rows = []

    # reading csv file
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        fields = next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)

        # get total number of rows
        print("Total no. of rows: %d" % (csvreader.line_num))

    # printing the field names
    print('Field names are:' + ', '.join(field for field in fields))

    check = ""
    #  printing first 5 rows
    for row in rows:
        if row[0] == name:
            check = name
    return check

if find_people("Hai Au") == "":
    print("unknown")