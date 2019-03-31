"""
Author: Cristina Mata
June 22, 2017
"""

def filter_list(source):
    """ Filter an image set list
    
    Arguments:
        source (string): The name of the file to filter. Every line must be in format: 'img_name int'.
    
    Returns:
        Creates file called filtered.txt, each line of which is an image name to be included in the set.
    """
    list_file = open(source, 'r')
    acc = []
    #results output in a new file called filtered.txt
    with open('filtered.txt', 'xt') as filtered:
        for line in list_file:
            split = line.split()
            #check if integer value is 1. If so, include image name in output.
            if split[1] == '1':
                n = split[0] + '\n'
                acc.append(n)
                filtered.write(n)
        filtered.close()
    print(len(acc))

#change name of file to filter here
filter_list('person_val.txt')
