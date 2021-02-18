from typing import Optional
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class _FullNormUpdater(nn.Function):

    @staticmethod
    def forward(ctx, X, fullnorm):
        # A will be of shape (N, N) and X may be of dimension (B, N) or (A, B, N).
        self.fullnorm = fullnorm
        if fullnorm.A.device != X.device:
            fullnorm.A = fullnorm.A.to(X.device)

        X = X.detach()
        A = fullnorm.A  # note: A has no grad.
        XA = torch.matmul(X, A)
        ctx.save_for_backward(XA)
        return XA

    @staticmethod
    def backward(ctx, out_grad):
        # In the backward pass we update fullnorm.A.   Only the grad w.r.t. X is
        # returned (A does not participate in the autograd, it is updated by formula).

        fullnorm = self.fullnorm
        # A is symmetric so there is no need to transpose it.
        X_grad = torch.matmul(out_grad, fullnorm.A)
        if fullnorm.rate != 0:
            # Update fullnorm.A
            A = fullnorm.A
            dim = A.shape[0]
            XA = ctx.saved_tensors()  # Note, X has no grad.
            XA = XA.reshape(-1, dim)
            XAA = torch.matmul(XA, A)
            # If our aim was to make X completely "white", we'd be updating with AXXA, but
            # we want X only partially decorrelated (halfway there), so we'll use AAXXAA.
            AAXXAA = torch.matmul(XAA, XAA.t()) * (1.0 / dim)
            # delta_A will be zero (i.e. we will be at equilibrium) when A ** 4 == Cov,
            # where Cov == torch.matmul(X,X.t()) / dim, and "** 4" means
            # A * A * A * A with '*' as matrix multiplication.
            # Note: if this equation were to read: A ** 2 == Cov, then we'd be making
            # the output X A have unit covariance.
            delta_A = torch.eye(dim, device=A.device) - AAXXAA
            A = A + delta_A * fullnorm.rate
            # make sure A stays fully symmetric even in the presence of roundoff.
            A = (A + A.t()) * 0.5

        return X_grad, None




class FullNorm(nn.Module):
    """
    FullNorm is a kind of normalization that aims to get the full covariance matrix
    of the input closer to the identity (in fact, it normalizes only about halfway,
    i.e. divides by sqrt(cov)).

    It owns a matrix-- a symmetric matrix that it multiplies the input by-- but this
    is not treated as a parameter, i.e. it is not learned with gradient descent.
    Instead, the 'backward' function updates the matrix using a formula that depends
    on the inputs but not on the gradient w.r.t. the output.

    The matrix approaches 1/sqrt(cov) only gradually, and as far as the rest of the
    network is concerned, this matrix is just a fixed parameter, i.e. we don't
    expose the derivative w.r.t. this adjustment as batchnorm does.  (In fact, because
    the adjustment is slow, this would be hard to do).
    """
    def __init__(self, feat_dim, rate = 5.0e-04):
        '''
        'rate' is like a learning rate-- it represents approximately how many minibatches it would take
        to get halfway to equilibrium.
        '''
        super(self, FullNorm).__init__()

        # note: we don't set self.A = nn.Parameter(torch.eye(feat_dim)); self.A
        # is not trainable in the normal way.
        self.register_buffer('A', torch.eye(feat_dim))
        self.rate = rate


    def forward(self, X):
        '''
        Forward of this module: multiplies X by self.A.  Equivalent to:
           `return torch.matmul(X, self.A)`,
        except it takes care of updating self.A if there is a backward pass.
        '''
        return _FullNormUpdater.apply(X, self)
