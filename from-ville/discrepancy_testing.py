#Tests with GP as utility/probability weighting function
import numpy as np
import torch
import matplotlib.pyplot as plt
import GPy
import gpytorch as gpt
from torch import nn, optim
torch.manual_seed(1213)
torch.set_default_dtype(torch.float64)

class CPT_distorted(nn.Module):
    def __init__(self, alpha, beta, lamb, gamma_pos, gamma_neg, f, K_L_inv, X, kernel):
        super(CPT_distorted, self).__init__()

        #non_restricted_params optimizable
        # self.alpha_n = nn.Parameter(torch.Tensor([alpha])) #positive reward power. < 1
        # self.beta_n = nn.Parameter(torch.Tensor([beta])) #negative reward power. < 1
        # self.lamb_n = nn.Parameter(torch.Tensor([lamb])) #negative reward multiplier (loss aversion parameter). > 1, more likely to be 2-2.5?
        # self.gamma_pos_n = nn.Parameter(torch.Tensor([gamma_pos]))
        # self.gamma_neg_n = nn.Parameter(torch.Tensor([gamma_neg]))
        #params not optimizable
        self.alpha_n = torch.Tensor([alpha]) #positive reward power. < 1
        self.beta_n = torch.Tensor([beta]) #negative reward power. < 1
        self.lamb_n = torch.Tensor([lamb]) #negative reward multiplier (loss aversion parameter). > 1, more likely to be 2-2.5?
        self.gamma_pos_n = torch.Tensor([gamma_pos])
        self.gamma_neg_n = torch.Tensor([gamma_neg])

        self.f = nn.Parameter(f.clone().detach()) #these should probably be parameters?
        #self.f = f.clone().detach().requires_grad_(True)#from warning, not working
        self.K_L_inv = K_L_inv
        self.X = X
        self.kernel = kernel
        self.map_params()
    
    def map_params(self):
        #map all of the parameters to the correct subspace
        self.alpha = torch.sigmoid(self.alpha_n) #positive reward power. < 1
        self.beta = torch.sigmoid(self.beta_n) #negative reward power. < 1
        #self.lamb = torch.exp(self.lamb_n) #negative reward multiplier (loss aversion parameter). > 1, more likely to be 2-2.5?
        sp = nn.Softplus()
        self.lamb = sp(self.lamb_n)
        self.gamma_pos = torch.sigmoid(self.gamma_pos_n)
        self.gamma_neg = torch.sigmoid(self.gamma_neg_n)
        # self.gamma_pos = sp(self.gamma_pos_n)
        # self.gamma_neg = sp(self.gamma_neg_n)


    def forward(self, X, p_seq, debug=False):
        self.map_params()

        w_neg_diff1 = self.w_neg(p_seq[1:]) - self.w_neg(p_seq[:-1])
        w_pos_diff1 = self.w_pos(1 - p_seq[1:]) - self.w_pos(1 - p_seq[:-1])
        w_neg_diff = torch.cat((self.w_neg(p_seq[0].unsqueeze(0)), w_neg_diff1), dim=0)
        w_pos_diff = torch.cat((self.w_pos(1-p_seq[0]) - 1, w_pos_diff1), dim=0)

        pos_mask = X >= 0
        UX = self.utility1(X)
        neg_sum = torch.matmul( ((~pos_mask).float() * UX).transpose(1,2),  w_neg_diff.unsqueeze(1).repeat(X.shape[0], 1, 1))
        pos_sum = torch.matmul( (pos_mask.float() * UX).transpose(1,2), w_pos_diff.unsqueeze(1).repeat(X.shape[0], 1, 1))
        CPVs = neg_sum - pos_sum
        return CPVs.view(-1)
    
    #gains probability weighing
    def w_pos(self, p):
        return p**self.gamma_pos / (p**self.gamma_pos + (1 - p)**self.gamma_pos)**(1/self.gamma_pos)
    #losses probability weighing
    def w_neg(self, p):
        return p**self.gamma_neg / (p**self.gamma_neg + (1 - p)**self.gamma_neg)**(1/self.gamma_neg)

    def distortion(self, outcome):
        with torch.no_grad():
            K_stars = self.kernel(self.X.repeat(outcome.shape[0],1,1),  outcome).evaluate().transpose(1,2)
            
        Kinv_times_f = self.K_L_inv.T @ self.K_L_inv @ self.f
        
        f_star_means = torch.matmul(K_stars, Kinv_times_f.unsqueeze(0) )
        # check computation
        # check_i = 0
        # check = K_stars[check_i] @ Kinv_times_f
        # (check - f_star_means[check_i]).abs().max() < 0.0001
        return f_star_means

    def utility1(self, outcome):
        ret = torch.zeros_like(outcome)
        pos_inds = outcome > 0
        ret[pos_inds] = outcome[pos_inds].pow(self.alpha)
        ret[~pos_inds] = - self.lamb * (- outcome[~pos_inds]).pow(self.beta)
        dd = self.distortion(outcome)
        return ret + dd

    def get_params(self, unconstrained=False):
        if unconstrained:
            return (self.alpha_n.detach(), self.beta_n.detach(), self.lamb_n.detach(), self.gamma_pos_n.detach(), self.gamma_neg_n.detach())
        return (self.alpha.detach(), self.beta.detach(), self.lamb.detach(), self.gamma_pos.detach(), self.gamma_neg.detach())


N_train = 300
xx = torch.linspace(-10,10, N_train).unsqueeze(1)

#sample from GP
sigma_ker = 0.2
len_ker = 0.9
#rbf_kern = GPy.kern.RBF(1, variance=sigma_ker, lengthscale=len_ker)
base_kern = gpt.kernels.RBFKernel(ard_num_dims=1)
base_kern.lengthscale=len_ker
rbf_kern = gpt.kernels.ScaleKernel(base_kern)
rbf_kern.outputscale = sigma_ker
jitter = 0.000001
with torch.no_grad():
    K = rbf_kern(xx, xx).evaluate()

L = torch.cholesky(K + jitter * torch.eye(xx.shape[0]))
L_inv = torch.inverse(L)
K_inv = torch.inverse(K)

z = torch.randn((N_train,1))
f = L @ z

#predict the grid
# xxx = torch.linspace(-5,5, 500).reshape(-1, 1)
# asd = rbf_kern(xx, xxx).evaluate()
# K_star = rbf_kern(xx, xxx).evaluate().T
# f_star = K_star @ K_inv @ f
# f_star_stable = K_star @ L_inv.T @ L_inv @ f
# torch.abs(f_star - f_star_stable).max()
# plt.plot(xx, f.detach())
# #plt.plot(xxx, f_star.detach())
# plt.plot(xxx, f_star_stable.detach())

#plot utility and discrepancy
gen_params = torch.randn(5)
gen_params[2] = gen_params[2].abs() + 1
gen_CPT = CPT_distorted(*gen_params, f = f, K_L_inv = L_inv, X  = xx, kernel = rbf_kern)
#gen_CPT.utility1(xx)
xxx = torch.linspace(-10,10, 1000)
xxx = xxx.reshape(1, -1, 1)
plt.plot(xxx.squeeze(0), gen_CPT.utility1(xxx).detach().squeeze(0))
plt.hlines(0, -5, 5)
plt.vlines(0, -2, 2)

#generate some lottery sets
N_l = 2*2**12
N_options = torch.randint(3, 5, (N_l,))
p_seq_dim = 50
p_seq = torch.linspace(0.001, 0.999, p_seq_dim)
#plt.hist((torch.distributions.Beta(1, 1).sample((1000,)) - 0.5) * 18)
((torch.distributions.Beta(1, 1).sample((1000,)) - 0.5) * 20).max()
rvs = [torch.distributions.Normal((torch.distributions.Beta(1, 1).sample((n,)) - 0.5) * 20, torch.distributions.Beta(0.6, 0.8).sample((n,))*2 + 0.1) for n in N_options]
quants = [rv_set.icdf(p_seq.repeat(rv_set.scale.shape[0], 1).T).T for rv_set in rvs]
#TODO try to reject lottery sets with no information

#generate_decicions
with torch.no_grad():
    decisions = [ torch.distributions.Categorical( gen_CPT(quant_set.unsqueeze(2), p_seq).softmax(0)).sample((1,)).squeeze(0) for quant_set in quants]#stochastic
    decisions = torch.tensor(decisions)

train_X = quants
train_y = decisions

#Plotting
# ii = torch.randint(100, (1,))
# quant_set = quants[ii]
# for aa in range(quant_set.shape[0]):
#     plt.plot(quant_set[aa], p_seq)

#try to optimize the GP mean
with torch.no_grad():
    b_k = gpt.kernels.RBFKernel(1)
    b_k.lengthscale = len_ker
    b_k.raw_lengthscale.requires_grad_(False)
    learn_kern = gpt.kernels.ScaleKernel(b_k)
    learn_kern.outputscale = sigma_ker
#x_learn = torch.Tensor([-2., -1., 1., 1.5]).reshape(-1, 1)
N_learn = 80
x_learn = torch.linspace(-9, 9, N_learn).reshape(-1, 1)
with torch.no_grad():
    K_learn = learn_kern(x_learn,x_learn).evaluate()
    L_learn = torch.cholesky(K_learn +  jitter * torch.eye(N_learn))
    K_inv_learn = torch.inverse(K_learn)
    L_inv_learn = torch.inverse(L_learn)

with torch.no_grad():
    initial_f = torch.zeros(N_learn, 1)
    #initial_f = f.detach()
    initial_f_ = initial_f.clone()

rec_CPT = CPT_distorted(*gen_params, f = initial_f, K_L_inv = L_inv_learn, X  = x_learn, kernel = learn_kern)
n_epochs = 130
batch_size = 512
verbose = True
opt = optim.Adam(rec_CPT.parameters(), lr = 0.1)
#opt = optim.SGD(rec_CPT.parameters(), lr = 0.0001, momentum=0.9)
list(rec_CPT.parameters())
losses = []
print("TRAINING")
plt.close()
#K_inv_learning = rec_CPT.K_L_inv.T @ rec_CPT.K_L_inv #this need not to be repeated in training
for ii in range(n_epochs):
    if verbose: print(ii)
    epoch_ind_perm = torch.randperm(N_l)
    batch_per_epoch = int(np.floor(N_l / batch_size)) + 1
    ep_losses = []
    for kk in range(batch_per_epoch):
        batch_inds = torch.arange(kk*batch_size, np.min(((kk+1)*batch_size, N_l)) ).long()
        batch_inds = epoch_ind_perm[batch_inds]
        actual_bs = batch_inds.shape[0]
        if batch_inds.shape[0] == 0:
            continue

        opt.zero_grad()
        # #handle the batch
        # #pad the batch with a sensible very negative sequence (this should not alter the probabilities)
        X_batch = [train_X[bi] for bi in batch_inds]
        y_batch = train_y[batch_inds]
        padding_sequence =  torch.linspace(0,1, X_batch[0].shape[1]) - 99999.0
        padding_sequence.unsqueeze_(0)
        #adapted from pad_sequence pytorch code, the only diff is that the padding is now a sequence
        quantile_dim = X_batch[0].size()[1]
        max_len = max([s.size(0) for s in X_batch])
        padded_batch = padding_sequence.repeat(actual_bs, max_len, 1)
        for i, tensor in enumerate(X_batch):
            length = tensor.size(0)
            padded_batch[i, :length, ...] = tensor
        #forward pass on padded and flattened to avoid complications
        padded_batch_flatted = padded_batch.view(-1, p_seq_dim, 1)
        #padded_batch_flatted.reshape(32, 9, 50) == padded_batch
        preds = rec_CPT(padded_batch_flatted, p_seq)
        #transform back
        preds = preds.view(actual_bs, max_len)
        pred_probs = preds.log_softmax(1)
        # check_i = 3
        # X_batch[check_i].shape
        # pred_probs[check_i]
        y_pred_probs = torch.gather(pred_probs, 1, index=y_batch.unsqueeze(1))
        log_prior = - 0.5 * rec_CPT.f.T @ rec_CPT.K_L_inv.T @ rec_CPT.K_L_inv @ rec_CPT.f
        # print(log_prior)
        # print(y_pred_probs.log().sum(0))
        loss = - (y_pred_probs.sum(0) + 0.1 * actual_bs / N_l * log_prior) #Hoffman and Blei?

        # loss = 0
        # for bi in batch_inds:
        #     X = train_X[bi]
        #     y = train_y[bi]
        #     preds = rec_CPT(X, p_seq)
        #     pred_probs = preds.softmax(0)
        #     loss += - torch.log(pred_probs[y])

        ep_losses.append(loss.detach().item())
        loss.backward()
        #nn.utils.clip_grad_norm_(rec_CPT.parameters(), max_norm=1.0)
        print(list(rec_CPT.parameters())[0].grad.abs().mean())
        opt.step()
         
    if verbose: print(np.mean(ep_losses))
    losses.append(np.mean(ep_losses))
    plt.plot(xx, f.detach())
    plt.plot(x_learn, initial_f_)
    plt.plot(x_learn, rec_CPT.f.detach())
    plt.savefig("plots/"+str(ii)+".jpg")
    plt.ylim(-2,2)
    plt.close()

plt.plot(losses)
plt.show()
plt.plot(losses[10:])
plt.show()

plt.plot(xxx.squeeze(0), gen_CPT.utility1(xxx).squeeze(0).detach() -  gen_CPT.distortion(xxx).squeeze(0).detach(), color="black")
plt.plot(xxx.squeeze(0), gen_CPT.utility1(xxx).squeeze(0).detach(), color="red")
plt.hlines(0, -5, 5)
plt.vlines(0, -2, 2)
plt.scatter(x_learn, rec_CPT.utility1(x_learn.reshape(1,-1,1)).squeeze(0).detach(), color="red")
#plt.plot(x_learn, rec_CPT.utility1(x_learn.reshape(1,-1,1)).squeeze(0).detach(), color="pink")
plt.scatter(x_learn, rec_CPT.utility1(x_learn.reshape(1,-1,1)).squeeze(0).detach() - rec_CPT.distortion(x_learn.reshape(1,-1,1)).squeeze(0).detach() + initial_f_, color="purple" ) #initial distortion

#in the distortion only
plt.scatter(xx, f.detach())
plt.plot(x_learn, initial_f_)
plt.scatter(x_learn, rec_CPT.f.detach())

plt.plot(xx, f.detach())
plt.plot(x_learn, initial_f_)
plt.plot(x_learn, rec_CPT.f.detach())

