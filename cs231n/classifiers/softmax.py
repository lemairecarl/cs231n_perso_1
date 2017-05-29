import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    f_i = X[i].dot(W)  # (1,D) (D,C) => (1,C)
    f_i -= np.max(f_i)  # For numerical stability
    scores_i_exp = np.exp(f_i)
    sum_exp = np.sum(scores_i_exp)
    class_probabilities = scores_i_exp / sum_exp  # softmax
    L_i = -np.log(scores_i_exp[y[i]] / sum_exp)
    loss += L_i

    dLi_dfi = class_probabilities - (range(num_classes) == y[i])
    dW += X[i][:, None].dot(dLi_dfi[None, :])

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = X.dot(W)
  f -= np.max(f, axis=1)[:, None]  # For numerical stability
  exp_f = np.exp(f)
  sum_exp_f = np.sum(exp_f, axis=1)[:, None]
  p = exp_f / sum_exp_f  # Class probabilities
  correct_class_mask = np.arange(num_classes)[None, :] == y[:, None]
  loss_i = -f[correct_class_mask][:, None] + np.log(sum_exp_f)
  loss = np.sum(loss_i) / num_train

  dL_dLi = 1.0 / num_train
  dLi_f = p - correct_class_mask
  dL_df = dL_dLi * dLi_f
  dW = X.T.dot(dL_df)

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

