# Create train/val/test lists where inputs are encodings instead of images/labels

def create_encoding_list_train(trainfile):
    names = []
    with open(trainfile, 'r') as f:
        for line in f:
            names.append("/encoder_output/"+line[12:23]+".npy\n")

    with open('train_encoding.txt', 'w') as wf:
        for name in names:
            wf.write("%s" % name)
    wf.close()

def create_encoding_list_val(valfile):
    names = []
    with open(valfile, 'r') as f:
        for line in f:
            names.append("/encoder_output/"+line[12:23]+".npy\n")

    with open('val_encoding.txt', 'w') as wf:
        for name in names:
            wf.write("%s" % name)
    wf.close()

def create_encoding_list_test(testfile):
    names = []
    with open(testfile, 'r') as f:
        for line in f:
            names.append("/encoder_output/"+line[12:23]+".npy\n")

    with open('test_encoding.txt', 'w') as wf:
        for name in names:
            wf.write("%s" % name)
    wf.close()

if __name__ == "__main__":
    create_encoding_list_test("test.txt")
