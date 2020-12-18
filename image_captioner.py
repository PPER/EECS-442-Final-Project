import torch
import torch.nn as nn
from word_embed import *
from feature_extractor import *
from attention import *
from util import *


class ImageCaptioner(nn.Module):
    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, device='cpu',
                 ignore_index=None, dtype=torch.float32):
        ### initialize ###
        super().__init__()
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)
        self.ignore_index = ignore_index

        ### model construction ###
        self.device = torch.device(device)
        self.dtype = dtype

        self.word_embed = WordEmbedding(vocab_size=vocab_size, embed_size=wordvec_dim, device=device, dtype=dtype)
        self.hidden_to_score = nn.Linear(in_features=hidden_dim, out_features=vocab_size, bias=True).to(self.device,
                                                                                                        dtype=dtype)
        self.cnn_to_h0 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True).to(self.device,
                                                                                                 dtype=dtype)
        self.feat_extractor = FeatureExtractor(pooling=False, device=device, dtype=self.dtype)
        self.lstm_att = AttentionLSTM(input_size=wordvec_dim, hidden_size=hidden_dim, device=device, dtype=dtype)

    def forward(self, images, captions):
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        loss = 0.0
        features = self.feat_extractor.extract_mobilenet_feature(img=images)
        embedded_words = self.word_embed(captions_in)

        A = self.cnn_to_h0(features.permute(0, 3, 2, 1))
        A = A.permute(0, 3, 2, 1)
        hn = self.lstm_att(embedded_words, A)
        scores = self.hidden_to_score(hn)
        loss = temporal_softmax_loss(scores, captions_out, ignore_index=self.ignore_index)
        return loss

    def sample(self, images, max_length=15):
        N = images.shape[0]
        captions = self._null * images.new(N, max_length).fill_(1).long()
        attn_weights_all = images.new(N, max_length, 4, 4).fill_(0).float()
        features = self.feat_extractor.extract_mobilenet_feature(img=images)

        A = self.cnn_to_h0(features.permute(0, 3, 2, 1))
        A = A.permute(0, 3, 2, 1)
        prev_h = prev_c = A.mean(dim=(2, 3))

        col = self.word_embed(self._start)
        W = col.shape[0]
        embedded_words = torch.ones(size=(N, W), device=self.device, dtype=self.dtype) * col

        for i in range(max_length):
            attn, attn_weights = dot_product_attention(prev_h, A)
            next_h, next_c = lstm_step_forward(embedded_words, prev_h, prev_c, self.lstm_att.Wx, self.lstm_att.Wh,
                                               self.lstm_att.b, attn, self.lstm_att.Wattn)
            attn_weights_all[:, i, :, :] = attn_weights
            prev_h = next_h
            prev_c = next_c
            scores = self.hidden_to_score(next_h)
            captions[:, i] = torch.argmax(scores, dim=1)
            embedded_words = self.word_embed(captions[:, i])
        return captions, attn_weights_all.cpu()
