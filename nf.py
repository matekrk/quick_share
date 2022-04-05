import os
import random
import math
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import svd
from sklearn.utils.extmath import randomized_svd

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')

class LinearBlock(nn.Module):
    def __init__(self, input_dimension):
        """Linear Invertible Block f_i(x) = A_i(x)+b_i
        Args:
            input_dimension (int): Number of dimensions in the input.
        """
        super(LinearBlock, self).__init__()
        self.A = nn.Parameter(torch.eye(input_dimension).to(device))
        self.b = nn.Parameter(torch.zeros((input_dimension)).to(device))
        self.update()

    def forward(self, x, ldj, reverse = False):
        if reverse:
            y = F.linear(x, weight=self.Ainv, bias=self.binv)
            ldj -= self.ldj
        else:
            y = F.linear(x, weight=self.A, bias=self.b)
            ldj += self.ldj
        return y, ldj

    def update(self):
        self.Ainv = torch.linalg.pinv(self.A)
        self.binv = - torch.matmul(self.Ainv, self.b)
        self.ldj = torch.slogdet(self.A.data)[1]

def test_linear_block():
    f = LinearBlock(15)
    if torch.cuda.is_available():
        f = f.cuda()
    x = torch.randn((100,15)).to(device)
    ldj = torch.zeros((x.size(0), 1)).to(device)

    y, ldjo = f(x, ldj)
    xi, ldji  = f(y, ldjo, True)

    torch.testing.assert_allclose(xi, x, atol=0.01, rtol=0.01)
    torch.testing.assert_allclose(ldji, ldjo, atol=0.01, rtol=0.01)

    del f, x, xi, y, ldj, ldjo, ldji


class NN(nn.Module):
  """Small neural network used to compute scale or translate factors.
  Args:
      input_dimension (int): Number of dimensions in the input.
      hidden_dimension (int): Number of dimensions in the hidden layers.
      output_dimensiion (int): Number of dimensions in the output.
      activation (bool): Use activation.
  """
  def __init__(self, input_dimension, hidden_dimension,
               output_dimension, activation=True):
    super(NN, self).__init__()

    self.activation = activation
    self.in_layer = nn.Linear(input_dimension, hidden_dimension)
    self.mid_layer1 = nn.Linear(hidden_dimension, hidden_dimension)
    self.mid_layer2 = nn.Linear(hidden_dimension, hidden_dimension)
    self.out_layer = nn.Linear(hidden_dimension, output_dimension)

  def forward(self, x):
    x = self.in_layer(x) 
    if self.activation:
        x = F.relu(x)
    x = self.mid_layer1(x)
    if self.activation:
        x = F.relu(x)
    x = self.mid_layer2(x)
    if self.activation:
        x = F.relu(x)
    x = self.out_layer(x)
    return x


class AdditiveCouplingBlock(nn.Module):
    """Additive Coupling Layer. 
    Split x into xA, xB and return (xA+t(xB), xB). t is NN.
    Args:
      input_dimension (int): Number of dimensions in the input.
      hidden_dimension (int): Number of dimensions in the hidden layers.
      alternate (bool): reverse xA and xB.
      activation (bool): use activation in t.
    """
    def __init__(self,input_dimension,
                 hidden_dimension, alternate, activation):
        super(AdditiveCouplingBlock, self).__init__()
        assert input_dimension%2 == 0
        self.translate = NN(input_dimension//2, hidden_dimension,
                            input_dimension//2, activation)
        self.alternate = alternate
        
    def forward(self, x, ldj, reverse=False):
        if self.alternate:
            xB, xA = x.chunk(2, dim=1)
        else:
            xA, xB = x.chunk(2, dim=1)

        t = self.translate(xB)
        if reverse:
            yA = xA - t
        else:
            yA = xA + t

        if self.alternate:
            y = torch.cat((xB, yA), dim=1)
        else:
            y = torch.cat((yA, xB), dim=1)
        
        ldj += 0
        return y, ldj

    def update(self):
        pass

def test_additive_block():
    f = AdditiveCouplingBlock(20, 30, True, True)
    if torch.cuda.is_available():
        f = f.cuda()
    x = torch.randn((100,20)).to(device)
    ldj = torch.zeros((x.size(0), 1)).to(device)

    y, ldjo = f(x, ldj)
    xi, ldji  = f(y, ldjo, True)

    torch.testing.assert_allclose(xi, x, atol=0.01, rtol=0.01)
    torch.testing.assert_allclose(ldji, ldjo, atol=0.01, rtol=0.01)

    del f, x, xi, y, ldjo, ldj, ldji


class AffineCouplingBlock(nn.Module):
    """Affine Coupling Layer. 
    Split x into xA, xB and return (s(xB)*xA+t(xB), xB). 
    Args:
      input_dimension (int): Number of dimensions in the input.
      hidden_dimension (int): Number of dimensions in the hidden layers.
      alternate (bool): reverse xA and xB.
      activation (bool): use activation in t.
    """
    def __init__(self, input_dimension, 
                 hidden_dimension, alternate, activation):
        super(AffineCouplingBlock, self).__init__()
        assert input_dimension%2 == 0
        self.st = NN(input_dimension//2, hidden_dimension,
                            input_dimension, activation)
        self.scale = nn.Parameter(torch.ones(input_dimension//2))
        self.alternate = alternate
        
    def forward(self, x, ldj, reverse=False):
        if self.alternate:
            xB, xA = x.chunk(2, dim=1)
        else:
            xA, xB = x.chunk(2, dim=1)

        st = self.st(xB)
        s, t = st[:, 0::2], st[:, 1::2]
        s = self.scale * torch.tanh(s)
        if reverse:
            ldj -= torch.sum(s, dim=1).view(-1, 1)
            s = torch.exp(-s)
            yA = s * (xA - t)
        else:
            # firstly take s (without exp to gradient)
            ldj += torch.sum(s, dim=1).view(-1, 1)
            # but take exp into consideration when scaling (we want to do exp(s) * xA + t)
            s = torch.exp(s)
            yA = s * xA + t

        if self.alternate:
            y = torch.cat((xB, yA), dim=1)
        else:
            y = torch.cat((yA, xB), dim=1)
        
        return y, ldj

    def update(self):
        pass

def test_affine_block():
    f = AffineCouplingBlock(2, 30, True, True)
    if torch.cuda.is_available():
        f = f.cuda()
    x = torch.randn((10,2)).to(device)
    ldj = torch.zeros((x.size(0), 1)).to(device)

    y, ldjo = f(x, ldj)
    xi, ldji  = f(y, ldjo, True)

    torch.testing.assert_allclose(xi, x, atol=0.01, rtol=0.01)
    torch.testing.assert_allclose(ldji, ldj, atol=0.01, rtol=0.01)

    del f, x, xi, y, ldjo, ldj, ldji

class LipschitzLinear(nn.Module):
    """Lipschitz Linear Function
    Args:
        in_features (int): Number of dimensions of the input.
        out_features (int): Number of dimensions of the output.
        coeff (float): Lipstchit constant between 0 and 1 stricly.
        activation (bool): Use activation.
    """
    def __init__(self, in_features, out_features, coeff=0.90):
        super(LipschitzLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.atol = None
        self.rtol = None
        self.coeff = coeff
        self.n_iterations = 10
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

        h, w = self.weight.shape
        self.register_buffer('scale', torch.tensor(0.))
        self.register_buffer('u', 
                             F.normalize(self.weight.new_empty(h).normal_(0, 1),
                                         dim=0))
        self.register_buffer('v', 
                             F.normalize(self.weight.new_empty(w).normal_(0, 1),
                                         dim=0))
        self.compute_weight(True, 1000)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def compute_weight(self, update=True, 
                       n_iterations=None, atol=None, rtol=None):
        
        n_iterations = (self.n_iterations if n_iterations is None 
                        else n_iterations)
        atol = self.atol if atol is None else atol
        rtol = self.rtol if rtol is None else atol

        if n_iterations is None and (atol is None or rtol is None):
            raise ValueError('Need one of n_iteration or (atol, rtol).')

        if n_iterations is None:
            n_iterations = 20000

        u = self.u
        v = self.v
        weight = self.weight
        if update:
            with torch.no_grad():
                itrs_used = 0.
                for _ in range(n_iterations):
                    old_v = v.clone()
                    old_u = u.clone()
                    v = F.normalize(torch.mv(weight.t(), u), dim=0, out=v)
                    u = F.normalize(torch.mv(weight, v), dim=0, out=u)
                    itrs_used = itrs_used + 1
                    if atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / (u.nelement()**0.5)
                        err_v = torch.norm(v - old_v) / (v.nelement()**0.5)
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                if itrs_used > 0:
                    u = u.clone()
                    v = v.clone()
                    self.u = u
                    self.v = self.v
        sigma = torch.dot(u, torch.mv(weight, v))
        with torch.no_grad():
            self.scale.copy_(sigma)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        return weight / factor

    def forward(self, x):
        weight = self.compute_weight(update=False, n_iterations = 5)
        return F.linear(x, weight, self.bias)


class LipschitzNN(nn.Module):
    """Small Lipschitz neural network.
    Args:
        input_dimension (int): Number of dimensions in the input.
        hidden_dimension (int): Number of dimensions in the hidden layers.
        coeff (float): Lipstchit constant between 0 and 1 stricly.
        activation (bool): Use activation.
    """
    def __init__(self, input_dimension, hidden_dimension, coeff, activation):
        super(LipschitzNN, self).__init__()
        self.activation = activation
        self.in_layer = LipschitzLinear(input_dimension,
                                        hidden_dimension, coeff)
        self.mid_layer1 = LipschitzLinear(hidden_dimension,
                                          hidden_dimension, coeff)
        self.mid_layer2 = LipschitzLinear(hidden_dimension,
                                          hidden_dimension, coeff)
        self.out_layer = LipschitzLinear(hidden_dimension,
                                         input_dimension, coeff)
        
    def forward(self, x):
        x = self.in_layer(x) 
        if self.activation:
            x = F.relu(x)
        x = self.mid_layer1(x)
        if self.activation:
            x = F.relu(x)
        x = self.mid_layer2(x)
        if self.activation:
            x = F.relu(x)
        x = self.out_layer(x)
        return x


class ResidualBlock(nn.Module):
    """Residual Atomic Block function.
    Takes x and ldj as the input and returns x+g(x) and ldj + logdetgrad(x)
    Args:
        input_dimension (int): Number of dimensions in the input.
        hidden_dimension (int): Number of dimensions in the hidden layers.
        coeff (float): Lipstchit constant between 0 and 1 stricly.
        activation (bool): Use activation.
    """
    def __init__(self, input_dimension, hidden_dimension, coeff, activation):
        super(ResidualBlock, self).__init__()
        self.g = LipschitzNN(input_dimension, 
                            hidden_dimension, 
                            coeff, 
                            activation)
        
    def forward(self, x, ldj, reverse = False):
        if reverse:
            y = self.inverse_fixed_point(x)
            # approximation (to speed up)
            ldj_out = ldj# - self.logdetgrad(x)
        else:
            y = x + self.g(x)
            ldj_out = ldj + self.logdetgrad(x)
        return y, ldj_out

    def inverse_fixed_point(self, y, atol=1e-5, rtol=1e-5):
        x, x_prev = y - self.g(y), y
        i = 0
        tol = atol + y.abs() * rtol
        while not torch.all(torch.abs(x - x_prev) / tol < 1):
            x, x_prev = y - self.g(x), x
            i += 1
            if i > 5000:
                break
        return x
    
    def logdetgrad(self, x):
        def poisson_sample(lamb, n_samples):
            return np.random.poisson(lamb, n_samples)

        def poisson_1mcdf(lamb, k, offset):
            if k <= offset:
                return 1.
            else:
                k = k - offset
            """P(n >= k)"""
            s = 1.
            for i in range(1, k):
                s += lamb**i / math.factorial(i)
            return 1 - np.exp(-lamb) * s

        def batch_jacobian(g, x):
            jac = []
            for d in range(g.shape[1]):
                jac.append(torch.autograd.grad(torch.sum(g[:, d]), 
                        x, 
                        create_graph=True)[0].view(x.shape[0], 1, x.shape[1]))
            return torch.cat(jac, 1)

        def batch_trace(M):
            return M.view(M.shape[0], -1)[:, ::M.shape[1] + 1].sum(1)

        n_samples = poisson_sample(2., 10)
        coeff_fn = lambda k: 1 / poisson_1mcdf(2., k, 10) * \
              sum(n_samples >= k - 5) / len(n_samples)
        x = x.requires_grad_(True)
        g = self.g(x)
        jac = batch_jacobian(g, x)
        logdetgrad = batch_trace(jac)
        jac_k = jac
        for k in range(2, 2):
            jac_k = torch.bmm(jac, jac_k)
            logdetgrad = (logdetgrad 
                        + (-1)**(k + 1) / k * coeff_fn(k) * batch_trace(jac_k))

        return logdetgrad.view(-1, 1)

    def update(self):
        for m in self.modules():
            if isinstance(m, LipschitzLinear):
                m.compute_weight(update=True, n_iterations=100)

def test_residual_block():
    f = ResidualBlock(1, 30, 0.5, True)
    if torch.cuda.is_available():
        f = f.cuda()
    f.update()
    x = torch.randn((10,1)).to(device)
    ldj = torch.zeros((x.size(0),1)).to(device)

    y, ldjo = f(x, ldj)
    xi, ldji  = f(y, ldjo, True)

    torch.testing.assert_allclose(xi, x, atol=0.01, rtol=0.01)
    torch.testing.assert_allclose(ldji, ldj, atol=0.01, rtol=0.01)
    del f, x, xi, y, ldjo, ldj, ldji


class NormalizingFlow(nn.Module):
    """Normalizing Flow class.
    Takes x as input and return F(x) and sldj. If F(y, reverse=True) is 
    called then it returns the inverse of F. 
    Args:
        input_dimension (int): Number of dimensions in the input.
        hidden_dimension (int): Number of dimensions in the hidden layers.
        coeff (float): Lipstchit constant between 0 and 1 stricly.
        activation (bool): Use activation.
    """
    def __init__(self, atomic='Linear', dimension=1, hidden_dimension=15, 
                num_steps=10, activation=True, coeff=0.9):
        super(NormalizingFlow, self).__init__()
        self.dimension = dimension
        if atomic == 'Linear':
            self.flows = [LinearBlock(input_dimension=dimension)
                            for depth in range(num_steps)]
        if atomic == 'AffineCoupling':
            self.flows = [AffineCouplingBlock(input_dimension=dimension,
                                            hidden_dimension=hidden_dimension,
                                            alternate = (depth%2==0),
                                            activation=activation)
                                            for depth in range(num_steps)]
        elif atomic == 'AdditiveCoupling':
            self.flows = [AdditiveCouplingBlock(input_dimension=dimension,
                                            hidden_dimension=hidden_dimension,
                                            alternate = (depth%2==0),
                                            activation=activation)
                                            for depth in range(num_steps)]
        elif atomic == 'Residual':
            self.flows = [ResidualBlock(input_dimension=dimension,
                                            hidden_dimension=hidden_dimension, 
                                            coeff=coeff,
                                            activation=activation)
                                            for depth in range(num_steps)]           
        self.flows = nn.ModuleList(self.flows)

    def forward(self, x, reverse=False):
        sldj = torch.zeros((x.size(0), 1)).to(device)
        if reverse:
            flows = reversed(self.flows)
        else:
            flows = self.flows
        
        for block in flows:
            x, sldj = block(x, sldj, reverse)
        return x, sldj

    def update(self):
        for block in self.flows:
            block.update()
    
def test_normalizing_flow():
    for block_type in ['Linear', 'AffineCoupling', 'AdditiveCoupling',
                    'Residual']:
        print('Testing: '+block_type+'...')
        f = NormalizingFlow(atomic=block_type, dimension=10)
        if torch.cuda.is_available():
            f = f.cuda()
        x = torch.randn((5,10)).to(device)
        ldj = torch.zeros((x.size(0), 1)).to(device)

        y, ldjo = f(x)
        xi, ldji  = f(y, True)

        torch.testing.assert_allclose(xi, x, atol=0.01, rtol=0.01)
        print(ldji+ldjo, ldj)
        torch.testing.assert_allclose(ldji+ldjo, ldj, atol=0.1, rtol=0.1)
        del f, x, xi, y, ldjo, ldj, ldji
        print('Test OK!')


class Demo_Gaussians():
    def __init__(self, batch_size):
        scale = 4.
        centers = [(-1, 0.5),(-0.35, -0.5), (0.35, -0.5), (1, 0.5)]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = np.random.randn(2)*1.2
            idx = np.random.randint(4)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        self.dataset = torch.Tensor(dataset)
    
    def give_tensor(self):
        return self.dataset

    def plot(self):
        plt.figure(figsize=(5,5))
        x  = self.give_tensor().numpy()
        plt.hist2d(x[:,0], x[:,1],
                    range=[[-5, 5], [-5, 5]], bins=100, cmap='inferno')
        plt.xticks([])
        plt.yticks([])
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.show()

def plot_2D(model, sample,  device, epoch):
    plt.clf()
    plt.figure(1, figsize=(7,7))
    plt.suptitle("Epoch "+str(epoch))
    plt.subplot(2,2,2)
    plt.title("$F^{-1}(Z_g)$")
    z = Variable(torch.randn((1000, 2)).to(device))
    x_gen, ldj = model(z,True)
    x_gen = x_gen.detach().cpu().numpy()
    plt.hist2d(x_gen[:,0], x_gen[:,1],
               range=[[-5, 5], [-5, 5]], bins=100, cmap='inferno')
    plt.xticks([])
    plt.yticks([])
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.subplot(2,2,1)
    plt.title("$X_r$")
    x = sample(1000).give_tensor()
    plt.hist2d(x[:,0].numpy(), x[:,1].numpy(),
               range=[[-5, 5], [-5, 5]], bins=100, cmap='inferno')
    plt.xticks([])
    plt.yticks([])
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.subplot(2,2,3)
    plt.title('$F(X_r)$')
    x = Variable(x.to(device))
    z, ldj = model(x)
    z = z.detach().cpu().numpy()
    plt.hist2d(z[:,0],z[:,1] , 
               range=[[-5, 5], [-5, 5]], bins=100, cmap='inferno')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,2,4)
    plt.title('$Z_g$')
    z = torch.randn((1000, 2))
    plt.hist2d(z[:,0].numpy(),z[:,1].numpy(), 
               range=[[-5, 5], [-5, 5]], bins=100, cmap='inferno')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()
    
def loss_function(z, logdetjac):
    """ Loss Function. 
    Takes z with shape (batch size, dimensions) and 
    logdetjac with shape (batch size, 1) as inputs. 
    It returns the negative log likelihood.
    """ 
    # change of formula p^(x) = |Jac_F(x)| * q(F(x))
    # min - E_(x from p) [log p^(x)] = min - E_(x from p) [logdetjac_F(x)] + log q(F(x))
    # q(F(x)) = 1/(2pi) * e ^ (-0.5 * ||F(x)||^2)
    N, d = z.shape
    loss = - torch.mean(logdetjac) + torch.mean(0.5 * torch.norm(z, p=2, dim=1)**2) - d/2 * torch.log(torch.as_tensor(2*math.pi))
    return loss

def test_loss_function():
    batch_size = 256
    z = torch.randn((batch_size, 25)).to(device)
    logdetjac = torch.rand((batch_size, 1)).to(device)
    loss1 = loss_function(z, logdetjac)
    loss2 = loss_function(z*2+1, 2*logdetjac)
    assert loss1 < loss2, "Loss1 should be lower than Loss2"
    #torch.testing.assert_allclose(loss_function(torch.zeros((256, 784)), 
    #                                            torch.zeros((256,1)))
    #                                ,720.44 , atol=0.1, rtol=0.1)
    del batch_size, z, logdetjac, loss1, loss2

def update_lr(optimizer, lr):
    lr = lr/5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer, lr

def train(model, x_sample, batch_size, device=device):
    # device = torch.device('cuda:0')
    model = model.to(device)
    n_epochs = 1000
    batch_size = 10000
    lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr = lr)
    for epoch in range(0, n_epochs+1):
        # torch.cuda.empty_cache()
        x = Variable(x_sample(batch_size).give_tensor()).to(device)
        z, logdetjac = model(x)
        if epoch %250==0:
            plot_2D(model,x_sample, device, epoch)
        loss = loss_function(z, logdetjac)
        loss.backward()
        optimizer.step()
        if epoch%5 ==0:
            model.update()
        if epoch %50 ==0:
            print('[%d/%d]: \tloss: %.3f \tlr: %.5f' % ((epoch), n_epochs, loss.data.item(), lr))
        if epoch in [20, 250, 500,750, 1000]:
            optimizer, lr = update_lr(optimizer, lr)

def test_train():
    for type_flow in ["Linear", "AdditiveCoupling", "AffineCoupling", "Residual"]:
        model = NormalizingFlow(type_flow, num_steps=4,  dimension = 2)
        train(model, Demo_Gaussians, 1000)

def main():
    #test_linear_block()
    #test_additive_block()
    #test_affine_block()
    #test_residual_block()
    test_normalizing_flow() #TODO: Fix Residual
    #test_loss_function()
    #test_train()

if __name__ == "__main__":
    main()