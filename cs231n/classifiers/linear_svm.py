import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  C = W.shape[1]

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    sum_incorrect = 0  # for gradient
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1

      if margin > 0:
        loss += margin
        sum_incorrect += 1  # for gradient
        dW[:, j] += X[i]
    dW[:, y[i]] += -sum_incorrect * X[i]


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(np.power(W, 2))
  dW += 0.5 * reg * 2.0 * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  delta = 1.0

  scores = X.dot(W)  # One line = one image
  correct_mask = np.arange(0, num_classes)[None, :] == y[:, None]
  correct_class_score = scores[correct_mask][:, None]
  inside = scores - correct_class_score + delta
  inside[correct_mask] = 0
  margins = np.maximum(0, inside)
  #margins[correct_mask] = 0
  loss = np.sum(margins) / num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(np.power(W, 2))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  dloss_dmargins = np.ones_like(margins) / num_train
  inside_gt_zero = (inside > 0)  # dmargins_dinside
  dmargins_dscores = inside_gt_zero - np.sum(inside_gt_zero, axis=1)[:, None] * correct_mask
  dloss_dscores = dloss_dmargins * dmargins_dscores
  dW = X.T.dot(dloss_dscores)

  dW += 0.5 * reg * 2.0 * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
