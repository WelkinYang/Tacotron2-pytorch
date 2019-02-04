from model.tacotron import Tacotron
from model.modules import Encoder, Decoder, PostNet, PostCBHG

def create_model(is_training=True):
    encoder = Encoder()
    decoder = Decoder()
    postnet = PostNet()
    post_cbhg = PostCBHG()

    model = Tacotron(
        encoder=encoder,
        decoder=decoder,
        postnet=postnet,
        post_cbhg=post_cbhg)
    if is_training:
        model.train()
    else:
        model.eval()
    return model