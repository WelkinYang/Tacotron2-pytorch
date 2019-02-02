from hparams import hparams as hp
char_list = __import__(hp.char_list)

def get_vocab_size():
    return len(char_list.char_to_id)

def text_to_sequence(text):
    seq = [char_list.char_to_id(char) for char in text.split(" ").strip()]
    seq.append(char_list.char_to_id('~'))
    return seq