# ## Logistic function (Logf)

# In[15]:


class Logf:

    @staticmethod
    def forward(t, param):
        unc_t0, unc_h1, unc_k = param

        t0 = safe_exp(unc_t0)
        h1 = safe_exp(unc_h1)
        k = safe_exp(unc_k)
        pred = h1 / (1 + safe_exp(-k * (t - t0)))
        return pred

    @classmethod
    def loss(cls, param):
        pred = cls.forward(data.age, param)
        return np.sum((data.height - pred) ** 2)