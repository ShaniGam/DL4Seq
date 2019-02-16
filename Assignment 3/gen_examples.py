import rstr
from random import randint


# Creating examples based on a regex
# reg - the regular expression
# output - the name of the output file
# examples_num - number of examples
def create_examples(reg, output, examples_num):
    with open(output, 'w+') as out:
        for i in range(examples_num):
            out.write(rstr.xeger(reg))
            out.write('\n')
    out.close()


# Creating the train/test file
# reg_pos - regex of positive example
# reg_neg - regex of negative example
# output - the name of the output file
# size - number of examples
def create_data(reg_pos, reg_neg, output, size):
    with open(output, 'w+') as out:
        for i in range(size):
            r = randint(0, 1)
            if r == 0:
                out.write('POS ' + rstr.xeger(reg_pos))
            else:
                out.write('NEG ' + rstr.xeger(reg_neg))
            out.write('\n')
    out.close()


if __name__ == '__main__':
    # regular expressions
    reg_pos = r'[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+'
    reg_neg = r'[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+'

    # create files for positive and negative examples
    create_examples(reg_pos, 'pos_examples', 500)
    create_examples(reg_neg, 'neg_examples', 500)

    # create train and test files
    create_data(reg_pos, reg_neg, 'train', 3000)
    create_data(reg_pos, reg_neg, 'test', 600)