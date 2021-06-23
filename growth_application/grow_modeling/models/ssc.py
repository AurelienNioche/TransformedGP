# ## Shohoji-Sasaki modified by Cole (SSC)

# T. Cole. The use and construction of anthropometric growth reference standards.
# Nutrition research reviews, 6(1):19–50, 1993.

# In[24]:


class Ssc:

    @staticmethod
    def forward(t, param):
        unc_h1, unc_k, beta0, beta1, unc_c, unc_r, unc_t_star = param

        h1 = safe_exp(unc_h1)
        k = safe_exp(unc_k)
        c = safe_exp(unc_c)
        r = safe_exp(unc_r)
        t_star = safe_exp(unc_t_star)

        Wt = safe_exp(-safe_exp(k * (t_star - t)))

        ft = beta0 + beta1 * t - safe_exp(c - r * t)
        pred = 0.1 * (h1 * Wt + ft * (1 - Wt))
        return pred

    @classmethod
    def loss(cls, param):
        pred = cls.forward(data.age, param)
        return np.sum((data.height - pred) ** 2)