if __name__ == "__main__":
    tot_names1 = []
    with open("train.txt", 'r') as f:
        for line in f:
            tot_names1.append(line)

    tot_names2 = []
    with open("val_classaug.txt", 'r') as f:
        for line in f:
            tot_names2.append(line)

    #print("train list ", sorted(tot_names1))
    #print("val list ", sorted(tot_names2))
    tot_names1 = set(tot_names1)
    tot_names2 = set(tot_names2)

    print("intersection of two lists ", tot_names1.intersection(tot_names2))
