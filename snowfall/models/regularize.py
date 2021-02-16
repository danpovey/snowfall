from typing import Optional, Any
import torch
from torch import nn
from torch import Tensor


class RegularizeGlobal:
    def __init__(self,
                 model: nn.Module,
                 scale: float = 1.0e-04,
                 interval: int = 10,
                 buffer_size: int = 10):
        self.model = model
        self.scale = scale
        self.interval = interval
        self.buffer_size = buffer_size
        self.buf = [None] * buffer_size


    def set_normal_mode(self):
        for module in self.network.modules():
            if isinstance(module, RegularizeLayer):
                module.set_normal_mode()




"""
This is a type of regularization that tries to prevent the network's activations
at selected locations (including, most probably, the output) from changing too
fast as we train.  The idea is that every K minibatches we penalize the sum-of-squared
difference between this minibatch's activations and the same minibatch evaluated
with the network aas it was (B*K) minibatches ago, where B is a buffer size (since
we need to store the activations in order to measure the difference).

In order to do this, we sometimes evaluate the neural network output for minibatches
just for the purpose of storing some intermediate activations (we don't want to
train on the example if we're about to evaluate activation differences on it).
"""

class RegularizeRecord:
    """
    RegularizeRecord is an object that we store in a rolling buffer of size B
    (buffer size).  We create a new one each time we want to store some activations,
    and we consume it B*K minibatches later.
    """
    def __init__(regularize_scale: float,
                 network: nn.Module,
                 batch: Any,
                 print_diffs: bool = False):
        """
        Create object that stores activations inside a neural network (wherever
        the RegularizeLayer module appears) and later uses the store activations
        to add a regularization term to the derivative that minimizes the change
        in those activations.

        Args:
           regularize_scale:
            Scale on the sum-of-squared-difference regularizer, e.g. 0.01.
            Regularizer will in effect be added to the loss function, times
            this scale (but only affects the derivative, not the loss value).
          network:
            The entire network, as a nn.Module (we'll locate modules of type
            RegularizerLayer in network.modules()).
        """
        self.regularize_scale = torch.Tensor(regularize_scale)
        self.network = network
        self.batch = batch

        for module in network.modules():
            if isinstance(module, RegularizeLayer):
                module.set_store_mode(self, pos)
        self.activation_list = []
        self.pos = 0  # we'll advance pos when we are in regularization mode later on.
        self.diffs = [] if print_diffs else None

    def set_regularize_mode(self) -> None:
        """
        Sets all RegularizeLayer objects within self.network to 'regularize mode' and sets their
        `regularize_record` member to self.
        """
        for module in network.modules():
            if isinstance(module, RegularizeLayer):
                module.set_regularize_mode(self)

    def set_store_mode(self) -> None:
        """
        Sets all RegularizeLayer objects within self.network to 'store mode' and sets their
        `regularize_record` member to self.
        """
        for module in network.modules():
            if isinstance(module, RegularizeLayer):
                module.set_store_mode(self)

    def store_activation(self, t: torch.Tensor) -> None:
        """
        In 'store' mode, when the RegularizeLayer does forward() it will call this.
        We just store the activation.
        """
        self.activation_list.append(t.detach().to(torch.half))

    def get_regularizer_deriv(self, t: torch.Tensor) -> torch.Tensor:
        """
        This function is used in 'regularize' mode.  Given a tensor t,
        which represents the current activations at a particular layer,
        it returns the derivative of the regularizer loss term (which is a
        scaled-sum-of-squared-difference) w.r.t. t.
        """
        assert self.pos < len(self.activation_list)
        ref_tensor = self.activation_list[self.pos]
        self.pos = self.pos + 1
        assert t.shape == ref_tensor.shape
        ref_tensor = ref_tensor.to(d.dtype)
        self.regularize_scale = self.regularize_scale.to(t.device)
        diff = (t.detach() - ref_tensor)
        if self.diffs is not None:
            self.diffs.append((diff ** 2).sum().item())
        t_deriv = 2.0 * self.regularize_scale * diff
        return t_deriv


    def __del__(self):
        if self.diffs is not None:
            print("Regularization: diffs are: ", self.diffs, ", sum = ", sum(self.diffs))


class _AddConstantToDeriv(nn.Function):
    @staticmethod
    def forward(ctx, x: Tensor, extra_deriv: Tensor) -> Tensor
        ctx.save_for_backward(extra_deriv)
        return x

    @staticmethod
    def backward(ctx, x_deriv: Tensor): tuple[torch.Tensor]
        extra_deriv, = ctx.saved_tensors
        return (x_deriv + extra_deriv),

class RegularizeLayer(nn.Module):
    def __init__(self) -> None:
        # regularize_record will be None or of type RegularizeRecord.
        self.regularize_record = None
        # possible modes are 'normal', 'store', 'regularize'.
        self.mode = 'normal'

    def set_store_mode(self, regularize_record):
        self.mode = 'store'
        self.regularize_record = regularize_record

    def set_regularize_mode(self, regularize_record):
        self.mode = 'regularize'
        self.regularize_record = regularize_record

    def set_normal_mode(self):
        self.mode = 'normal'
        self.regularize_record = None

    # for set_normal_mode, see the function outside of this class.

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == 'normal':
            return x
        elif self.mode == 'store':
            self.regularize_record.store_activation(x)
            return x
        else:
            assert self.mode == 'regularize'
            reg_deriv = self.regularize_record.get_regularizer_deriv(x)
            return _AddConstantToDeriv.apply(x, reg_deriv)


class AcousticModel(nn.Module):
    """
    AcousticModel specifies the common attributes/methods that
    will be exposed by all Snowfall acoustic model networks.
    Think of it as of an interface class.
    """

    # A.k.a. the input feature dimension.
    num_features: int

    # A.k.a. the output dimension (could be the number of phones or
    # characters in the vocabulary).
    num_classes: int

    # When greater than one, the networks output sequence length will be
    # this many times smaller than the input sequence length.
    subsampling_factor: int

    def write_tensorboard_diagnostics(
            self,
            tb_writer: SummaryWriter,
            global_step: Optional[int] = None
    ):
        """
        Collect interesting diagnostic info about the model and write to to TensorBoard.
        Unless overridden, logs nothing.

        :param tb_writer: a TensorBoard ``SummaryWriter`` instance.
        :param global_step: optional number of total training steps done so far.
        """
        pass
