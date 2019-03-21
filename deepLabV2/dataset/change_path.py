def remove_aug():
    names = []
    with open('val.txt', 'r') as f:
        for line in f:
            # Get rid of Aug in SegmentationClassAug
            names.append(line[:46]+line[49:])

    with open('val_new.txt', 'w') as wf:
        for name in names:
            wf.write("%s" % name)
    wf.close()

def add_jpg():
    names = []
    with open('voc2012_train.txt', 'r') as f:
        for line in f:
            # Add .jpg to end
            names.append(line[:-1] + ".jpg\n")

    with open('voc2012_train_new.txt', 'w') as wf:
        for name in names:
            wf.write("%s" % name)
    wf.close()

def add_dirs_train_voc():
    names = []
    with open('voc2012_train_old.txt', 'r') as f:
        for line in f:
            im_name = line[:-1]
            # Convert to /JPEGImages/im_name.jpg /SegmentationClassAug/im_name.png
            #names.append("/JPEGImages/"+im_name+".jpg /SegmentationClass/"+im_name+".png\n")
            # Convert to /JPEGImages/im_name.jpg /SegmentationClass_1D/im_name.png
            names.append("/JPEGImages/"+im_name+".jpg /SegmentationClass_1D/"+im_name+".png\n")

    with open('voc2012_train.txt', 'w') as wf:
        for name in names:
            wf.write("%s" % name)
    wf.close()

def add_dirs_train_person():
    names = []
    with open('person_train_id.txt', 'r') as f:
        for line in f:
            im_name = line[:-1]
            # Convert to /images_orig/im_name.jpg /labels_orig/im_name.png
            names.append("/images_orig/"+im_name+".jpg /labels_orig/"+im_name+".png\n")

    with open('person_train.txt', 'w') as wf:
        for name in names:
            wf.write("%s" % name)
    wf.close()

def add_dirs_train_horsecow():
    names = []
    with open('horsecow_train_id.txt', 'r') as f:
        for line in f:
            im_name = line[:-1]
            # Convert to /images_orig/im_name.jpg /labels_orig/im_name.png
            names.append("/images_orig/"+im_name+".jpg /labels_orig/"+im_name+".png\n")

    with open('horsecow_train.txt', 'w') as wf:
        for name in names:
            wf.write("%s" % name)
    wf.close()

def add_dirs_test_voc():
    names = []
    with open('voc2012_test_old.txt', 'r') as f:
        for line in f:
            im_name = line[:-1]
            # Convert to /JPEGImages/im_name.jpg
            names.append("/JPEGImages/"+im_name+".jpg\n")

    with open('voc2012_test.txt', 'w') as wf:
        for name in names:
            wf.write("%s" % name)
    wf.close()

def add_dirs_test_person():
    names = []
    with open('person_test_id.txt', 'r') as f:
        for line in f:
            im_name = line[:-1]
            # Convert to /images_orig/im_name.jpg /labels_orig/im_name.png
            names.append("/images_orig/"+im_name+".jpg /labels_orig/"+im_name+".png\n")

    with open('person_test.txt', 'w') as wf:
        for name in names:
            wf.write("%s" % name)
    wf.close()

def add_dirs_test_horsecow():
    names = []
    with open('horsecow_test_id.txt', 'r') as f:
        for line in f:
            im_name = line[:-1]
            # Convert to /images_orig/im_name.jpg /labels_orig/im_name.png
            names.append("/images_orig/"+im_name+".jpg /labels_orig/"+im_name+".png\n")

    with open('horsecow_test.txt', 'w') as wf:
        for name in names:
            wf.write("%s" % name)
    wf.close()

if __name__ == "__main__":
    add_dirs_test_horsecow()
