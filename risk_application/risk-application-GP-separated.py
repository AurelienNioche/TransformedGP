#!/usr/bin/env python
# coding: utf-8

# In[105]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scipy
from tqdm.notebook import tqdm
import dill

import torch
import torch.distributions as dist

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy, UnwhitenedVariationalStrategy
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood

from pymc3.gp.util import plot_gp_dist


# # Run

# In[42]:


u_model = u_pow
theta = 0.5
tau = 100
n = 200
data = generate_data(u=u_model, n=n, tau=tau, theta=theta)
data


# ## Good prior

# In[43]:


model = DiscrepancyModel(
    data=data, u=u_model, theta=theta, tau=tau,  h=torch.sigmoid, n_samples=100,
    learn_inducing_locations=False, n_inducing_points=50)


# In[44]:


hist_loss = model.train(
    epochs=1000,
    learning_rate=0.1)


# In[46]:


plt.plot(hist_loss);


# In[47]:


test_x = torch.linspace(0, 1, 101)
f_preds = model.pred(test_x)


# In[48]:


fig, ax = plt.subplots()
for i in range(10):
    ax.plot(test_x, f_preds[i])


# In[49]:


fig, ax = plt.subplots()
plot_gp_dist(ax, f_preds.numpy(), test_x)
ax.plot(test_x, u_model(test_x, theta));


# ## Wrong prior

# In[50]:


model = DiscrepancyModel(
    data=data, 
    u=u_lin,  # Here is the change!
    theta=theta, tau=tau,  h=torch.sigmoid, n_samples=100,
    learn_inducing_locations=False, n_inducing_points=50)


# In[51]:


hist_loss = model.train(
    epochs=1000,
    learning_rate=0.1)


# In[53]:


plt.plot(hist_loss);


# In[54]:


test_x = torch.linspace(0, 1, 101)
f_preds = model.pred(test_x)


# In[55]:


fig, ax = plt.subplots()
plot_gp_dist(ax, f_preds.numpy(), test_x)
ax.plot(test_x, u_model(test_x, theta))
ax.plot(test_x, u_lin(test_x, None), ls=':', color='C2');


# # Using CPC dataset

# ## Generate artificial data

# In[12]:


def generate_data_cpc_like(u, seed=123, n=100, tau=3.333, theta=0.5):

    np.random.seed(seed)

    data = pd.DataFrame(np.zeros((n*10, 8)),
                        columns=[f"p{i}" for i in range(4)] + [f"x{i}" for i in range(4)])

    j = 0
    for opt in range(2):
        p = np.random.random(size=n*10)

        data[f'p{j}'] = p
        data[f'p{j+1}'] = 1 - p

        a = np.random.random(size=(n*10, 2))
        a = np.sort(-a, -1)*(-1)

        data[f'x{j}'] = a[:, 0]
        data[f'x{j+1}'] = a[:, 1]

        j += 2

    data = data[~((data.p0 >= data.p2) & (data.x0 >= data.x2))]
    data = data[~((data.p2 >= data.p0) & (data.x2 >= data.x0))]
    data = data.sample(n=n, replace=False)

    pA = data.p0.values
    pB = data.p2.values

    xA0 = data.x0.values
    xA1 = data.x1.values

    xB0 = data.x2.values
    xB1 = data.x3.values

    seuA = pA * u(xA0, theta) + (1-pA) * u(xA1, theta)
    seuB = pB * u(xB0, theta) + (1-pB) * u(xB1, theta)

    diff_eu = seuB - seuA

    p_chooseB = scipy.special.expit(tau * diff_eu)
    choices = np.zeros(n, dtype=int)
    choices[:] = p_chooseB > np.random.random(size=n)
    data['choices'] = choices

    return data


# ## Fit the data with 'expert' model

# In[13]:


def prelec(p, theta): return np.exp(-(-np.log(p))**theta)



# ## Run with artificial data

# ### Generate artificial data

# In[55]:


u_model = u_pow
tau = 100
theta = 0.5
n = 200
data = generate_data_cpc_like(u=u_model, tau=tau, theta=theta, n=n)
data


# In[56]:


opt_param = optimize(data, u=u_model, w=None)
tau_opt = opt_param[0]
theta_opt = opt_param[1]
print("tau", tau_opt, "theta_u", theta_opt)


# ### Run with good model

# In[57]:


model = DiscrepancyModel(
    data=data, 
    u=u_model, 
    theta=theta, tau=tau,  h=torch.sigmoid, n_samples=100,
    learn_inducing_locations=False, n_inducing_points=50)


# In[58]:


hist_loss = model.train(
    epochs=1000,
    learning_rate=0.1)


# In[59]:


plt.plot(hist_loss);


# In[60]:


test_x = torch.linspace(0, 1, 100)
f_preds = model.pred(test_x)


# In[61]:


fig, ax = plt.subplots()
plot_gp_dist(ax, f_preds.numpy(), test_x)
ax.plot(test_x, u_model(test_x, theta));


# ### Run with wrong model

# In[62]:


model = DiscrepancyModel(
    data=data, 
    u=u_lin,  # Here is the change!
    theta=theta, tau=tau,  h=torch.sigmoid, n_samples=100,
    learn_inducing_locations=False, n_inducing_points=50)


# In[63]:


hist_loss = model.train(
    epochs=1000,
    learning_rate=0.1)


# In[64]:


plt.plot(hist_loss);


# In[65]:


test_x = torch.linspace(0, 1, 100)
f_preds = model.pred(test_x)


# In[66]:


fig, ax = plt.subplots()
plot_gp_dist(ax, f_preds.numpy(), test_x)
ax.plot(test_x, u_model(test_x, theta))
ax.plot(test_x, u_lin(test_x, None), ls=':', color='C2');


# ## Run with CPC dataset

# In[67]:


df = pd.read_csv("../data/cpc2018.csv")


# In[68]:


data = df[(df.LotNumB == 1) & (df.LotNumA == 1) & (df.Amb == 0) 
          & (df.Ha >= 0) & (df.Hb >= 0) & (df.La >= 0) & (df.Lb >= 0)]  # & (df.La ==0)  & (df.Lb == 0)


# In[69]:


data = pd.DataFrame({
    "subject": data.SubjID,
    "p0": data.pHa.values,
    "x0": data.Ha.values,
    "p1": 1 - data.pHa.values,
    "x1": data.La.values,
    "p2": data.pHb.values,
    "x2": data.Hb.values,
    "p3": 1 - data.pHb.values,
    "x3": data.Lb.values,
    "choices": data.B.values
})
max_x = np.max(np.concatenate([data[f'x{i}'] for i in range(4)]))
for i in range(4):
    data[f'x{i}'] = data[f'x{i}'] / max_x


# In[70]:


data


# In[71]:


opt_param = optimize(data, u=u_model, w=None)
tau_opt = opt_param[0]
theta_opt = opt_param[1]
print("tau", tau_opt, "theta", theta_opt)


# In[72]:


fig, ax = plt.subplots()
ax.set_title(f"Population")
ax.plot(test_x, u_model(test_x, theta_opt))
plt.show()


# ### Try out with one subject

# In[76]:


s = 11303
d = data[data.subject == s]

opt_param = optimize(d, u=u_model, w=None)
tau = opt_param[0]
theta = opt_param[1]
print("tau", tau, "theta", theta)


# In[77]:


x = np.linspace(-1, 1, 100)
y = scipy.special.expit(tau_opt*x)
plt.plot(x, y);


# In[78]:


u = u_model

model = DiscrepancyModel(
    data=d, 
    u=u,
    theta=theta, 
    tau=tau,  
    h=torch.sigmoid, 
    n_samples=100,
    learn_inducing_locations=False, 
    n_inducing_points=50, 
    jitter=1e-07)

hist_loss = model.train(
    epochs=1000,
    learning_rate=0.1)


# In[79]:


plt.plot(hist_loss)


# In[80]:


test_x = torch.linspace(0, 1, 100)
f_preds = model.pred(test_x)

fig, ax = plt.subplots()

ax.set_title(f"Subject {s}; Model = '{u.__name__}'")
plot_gp_dist(ax, f_preds.numpy(), test_x)
ax.plot(test_x, u_model(test_x, theta_opt));


# In[81]:


u = u_lin

model = DiscrepancyModel(
    data=d, 
    u=u,
    theta=theta, 
    tau=tau,  
    h=torch.sigmoid, 
    n_samples=100,
    learn_inducing_locations=False, 
    n_inducing_points=50, 
    jitter=1e-07)

hist_loss = model.train(
    epochs=1000,
    learning_rate=0.1)


# In[85]:


plt.plot(hist_loss);


# In[90]:


test_x = torch.linspace(0, 1, 100)
f_preds = model.pred(test_x)

fig, ax = plt.subplots(figsize=(6, 3))

ax.set_title(f"Subject {s}; Model = '{u.__name__}'")
plot_gp_dist(ax, f_preds.numpy(), test_x)
ax.plot(test_x, u_model(test_x, theta_opt));


# ### Run with all subjects

# In[109]:


results = []

counts = data.subject.value_counts()
subject_325 =  counts[counts == 325].index

for s in tqdm(subject_325):
    
    try:
    
        d = data[data.subject == s]

        opt_param = optimize(d, u=u_model, w=None)

        tau = opt_param[0]
        theta = opt_param[1]
        print(f"Subject {s} tau={tau} theta={theta}")

        u = u_model

        model = DiscrepancyModel(
            data=d, 
            u=u,
            theta=theta, 
            tau=tau,  
            h=torch.sigmoid, 
            n_samples=100,
            learn_inducing_locations=False, 
            n_inducing_points=50, 
            jitter=1e-07)

        hist_loss = model.train(
            epochs=1000,
            learning_rate=0.1)

        test_x = torch.linspace(0, 1, 100)
        f_preds = model.pred(test_x)

        fig, ax = plt.subplots(figsize=(6, 3))

        ax.set_title(f"Subject {s}; Model = '{u.__name__}'")
        plot_gp_dist(ax, f_preds.numpy(), test_x)
        ax.plot(test_x, u_model(test_x, theta))

        plt.show()

        results.append({
            "subject": s, 
            "discrepancy_model": dill.dumps(model), 
            "hist_loss": hist_loss, 
            "f_preds": f_preds.numpy(),
            "utility_model": u.__name__,
            "tau": tau,
            "theta": theta
        })
    
    except Exception as e:
        print(f"Encountered error with subject {s}: {e}")


# In[110]:


pd.DataFrame(results).to_csv("results.csv")


# In[ ]:




