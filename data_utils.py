def get_vocab_size():
    return 100

def compute_same_padding(kernel_size, input_length, dilation=2):
    #when stride == 1, dilation == 2, groups == 1
    #Lout = [(Lin + 2 * padding - dilation * (kernel_size - 1) - 1) + 1]
    #padding = dilation * (kernel_size - 1) / 2
    return int(dilation * (kernel_size - 1) / 2)