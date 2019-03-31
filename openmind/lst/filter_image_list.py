if __name__ == "__main__":
    names = []
    with open('test.txt', 'r') as f:
        for line in f:
            names.append(line[13:24])

    with open('voc2012_test.txt', 'w') as wf:
        for name in names:
            wf.write("%s\n" % name)
    wf.close()
