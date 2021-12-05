# RMIT University Vietnam
# Course: COSC2429 Introduction to Programming
# Semester: 2020B
# Assignment: 3
# Author: La Tran Hai Dang (s3836605)
# Created date: 23/09/2020
# Last modified date: 23/09/2020


def task1():
    """
    The function to check the order and inventory
    """
    input_inventory = open("inventory.txt", "r")  # open the inventory file
    dict_inventory = {}  # creat the dictionary of the inventory
    line_inventory = input_inventory.readline()   # read each line of the inventory file

    while line_inventory:
        element_inventory = line_inventory.split()  # split the element in a line by white space
        for i in element_inventory:   # This loop to add the element in a line to dict_inventory
            dict_inventory[element_inventory[0]] = element_inventory[1]  # the first element is a key and the second element is a value
        line_inventory = input_inventory.readline()   # read the next line again
    input_order = open("order.txt", "r")  # open the order file
    line_order = input_order.readline()  # read each line of the order file
    while line_order:  # this while loop used to check the order
        element_order = line_order.split()  # split the element in order line file
        for value in element_order:    # the for loop to run due to the element in a line
            if value in dict_inventory:  # if the element in order have in dictionary
                for key in dict_inventory.keys():   # check in dictionary
                    if value == key:    # check if the value = keu in the dictionary
                        if int(element_order[2]) > int(dict_inventory.get(key)):   # check f the order is higher than current inventory
                            print(" The order", key, "in", element_order[0], "cannot be fulfilled due to out of stock")  # print the error
                        else:
                            new_value = int(dict_inventory.get(key)) - int(element_order[2])
                            dict_inventory[value] = new_value   # replace the value
                            print(" The order", key, "in", element_order[0], " be fulfilled")  # print the situation
        line_order = input_order.readline()

    # write and update the inventory
    outfile = open("inventory.txt", "w")
    for key in dict_inventory.keys():
        line = key + " " + str(dict_inventory.get(key)) + '\n' # the new line to update
        outfile.write(line)
    outfile.close()
    input_order.close()  # close the file
    input_inventory.close()  # close the file


def task2():
    """
        This function asks user to enter a filename to read as an input file. You can use file "order.txt" or "inventory.txt" above to test.
        The function then asks user to enter 2 numbers line1 and line2 on a single line, separated by white space.
    """
    # input the file
    file_name = str(input("Entering the filename to read: "))
    input_file = open(file_name, "r")
    new_output_file = open("newfile.txt", "w")
    # ask user to enter the lne number
    input_line = str(input("Entering  2 numbers line1 and line2 on a single line, separated by white space: "))
    count = 0
    line_list = []
    input_line.split()
    for i in input_line:  # this line to split the input of user
        if i != " ":
            line_list.append(i)  # append 2 line of user input in list
    if line_list[1] < line_list[0]:  # check the validity of 2 numbers line1 and line2
        print("ERROR line1 is larger than line2 ")
    elif line_list[1] > line_list[0]:
        for i in input_file:
            count += 1
            if count in range(int(line_list[0]), int(line_list[1]) + 1):
                new_output_file.write(i)
    new_output_file.close()
    input_file.close()


# to check the user choice
choice = str(input("Entering your choice 1(Task 1), 2(Task 2), 3(exit): "))
while (choice != "1") and (choice != "2") and (choice != "3"):  # if user enter out of the 3 choice
    print("Error")
    choice = str(input("Entering your choice 1,2,3 again: "))   # Enter the user choice again
if choice == "1":
    print(task1())
elif choice == "2":
    print(task2())
elif choice == "3":
    print("Good bye")
    exit()

