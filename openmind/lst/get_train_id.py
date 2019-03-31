"""
Processing image lists
"""

def get_list_id(source, new_name):
    list_file = open(source, 'r')
    with open(new_name, 'xt') as new_list:
        for line in list_file:
            new_list.write(line[13:24] + "\n")
        new_list.close()

if __name__ == "__main__":
    get_list_id("./list/train.txt", "./list/train_id.txt")
