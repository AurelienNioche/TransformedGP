# ## Jolicoeur, Pontier, Pernin and Sempé (Jpps)

# P. Jolicoeur, J. Pontier, M.-O. Pernin, and M. Sempé. A lifetime asymptotic
# growth curve for human height. Biometrics, pages 995–1003, 1988.

# In[21]:


class Jpps:

    @staticmethod
    def forward(t, param):
        unc_h1, unc_C1, unc_C2, unc_C3, unc_D1, unc_D2, unc_D3 = param

        h1 = safe_exp(unc_h1)
        C1 = safe_exp(unc_C1)
        C2 = safe_exp(unc_C2)
        C3 = safe_exp(unc_C3)
        D1 = safe_exp(unc_D1)
        D2 = safe_exp(unc_D2)
        D3 = safe_exp(unc_D3)

        tp = t + 0.75

        pred = h1 * (1 - 1 / (
                    1 + (tp / D1) ** C1 + (tp / D2) ** C2 + (tp / D3) ** C3))
        return pred

    @classmethod
    def loss(cls, param):
        pred = cls.forward(data.age, param)
        return np.sum((data.height - pred) ** 2)


# In[22]: