import torch
import torch.nn as nn
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2


def load_COCO(path = './datasets/coco.pt'):
  data_dict = torch.load(path)
  # print out all the keys and values from the data dictionary
  for k, v in data_dict.items():
      if type(v) == torch.Tensor:
          print(k, type(v), v.shape, v.dtype)
      else:
          print(k, type(v), v.keys())

  num_train = data_dict['train_images'].size(0)
  num_val = data_dict['val_images'].size(0)
  assert data_dict['train_images'].size(0) == data_dict['train_captions'].size(0) and \
        data_dict['val_images'].size(0) == data_dict['val_captions'].size(0), \
        'shapes of data mismatch!'

  return data_dict
  
def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != '<NULL>':
                words.append(word)
            if word == '<END>':
                break
        decoded.append(' '.join(words))
    if singleton:
        decoded = decoded[0]
    return decoded
    
def temporal_softmax_loss(x, y, ignore_index=None):
    loss = torch.nn.functional.cross_entropy(input=x.reshape(x.shape[0] * x.shape[1], x.shape[2]),
                                             target=y.reshape(y.shape[0] * y.shape[1]), ignore_index=ignore_index,
                                             reduction='sum') / x.shape[0]
    return loss
    
def captioning_train(rnn_model, image_data, caption_data, lr_decay=1, num_epochs=80, batch_size=50,
                learning_rate=1e-2):
  """
  Run optimization to train the model.
  """
  # optimizer setup
  from torch import optim
  optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, rnn_model.parameters()),
    learning_rate) # leave betas and eps by default
  lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                             lambda epoch: lr_decay ** epoch)

  # sample minibatch data
  iter_per_epoch = math.ceil(image_data.shape[0] // batch_size)
  loss_history = []
  rnn_model.train()
  for i in range(num_epochs):
    start_t = time.time()
    for j in range(iter_per_epoch):
      images, captions = image_data[j*batch_size:(j+1)*batch_size], \
                           caption_data[j*batch_size:(j+1)*batch_size]

      loss = rnn_model(images, captions)
      optimizer.zero_grad()
      loss.backward()
      loss_history.append(loss.item())
      optimizer.step()
    end_t = time.time()
    print('(Epoch {} / {}) loss: {:.4f} time per epoch: {:.1f}s'.format(
        i, num_epochs, loss.item(), end_t-start_t))

    lr_scheduler.step()

  # plot the training losses
  plt.plot(loss_history)
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.title('Training loss history')
  plt.show()
  return rnn_model, loss_history
  
def attention_visualizer(img, attn_weights, token):
  C, H, W = img.shape
  assert C == 3, 'We only support image with three color channels!'

  # Reshape attention map
  attn_weights = cv2.resize(attn_weights.data.numpy().copy(),
                              (H, W), interpolation=cv2.INTER_NEAREST)
  attn_weights = np.repeat(np.expand_dims(attn_weights, axis=2), 3, axis=2)

  # Combine image and attention map
  img_copy = img.float().div(255.).permute(1, 2, 0
    ).numpy()[:, :, ::-1].copy()  # covert to BGR for cv2
  masked_img = cv2.addWeighted(attn_weights, 0.5, img_copy, 0.5, 0)
  img_copy = np.concatenate((np.zeros((25, W, 3)),
    masked_img), axis=0)

  # Add text
  cv2.putText(img_copy, '%s' % (token), (10, 15),
              cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=1)

  return img_copy