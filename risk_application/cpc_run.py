import torch

from cognitive_modeling.models.utility_models import u_pow, u_lin
from simulation.cpc_like import generate_data_cpc_like
from cognitive_modeling.cpc_like import fit_cpc_like
from discrepancy_modeling.discrepancy_modeling import DiscrepancyModel


def run_artificial():

    u_model = u_pow
    tau = 100
    theta = 0.5
    n = 200

    print("generating data...")
    data = generate_data_cpc_like(u=u_model, tau=tau, theta=theta, n=n)

    print("fitting data...")
    opt_param = fit_cpc_like(data, u=u_model, w=None)
    tau_opt = opt_param[0]
    theta_opt = opt_param[1]
    print("tau", tau_opt, "theta_u", theta_opt)


    # Run with good model
    model = DiscrepancyModel(
        data=data,
        u=u_model,
        theta=theta, tau=tau,  h=torch.sigmoid, n_samples=100,
        learn_inducing_locations=False, n_inducing_points=50)

    test_x = torch.linspace(0, 1, 100)
    f_preds = model.pred(test_x)


    # In[61]:
    from pymc3.gp.util import plot_gp_dist


    fig, ax = plt.subplots()
    plot_gp_dist(ax, f_preds.numpy(), test_x)
    ax.plot(test_x, u_model(test_x, theta))
    plt.show()

#
#     # In[58]:
#
#
#     hist_loss = model.train(
#         epochs=1000,
#         learning_rate=0.1)
#
#
#     # In[59]:
#
#
#     plt.plot(hist_loss);
#
#
#
# # ### Run with wrong model
#
# # In[62]:
#
#
# model = DiscrepancyModel(
#     data=data,
#     u=u_lin,  # Here is the change!
#     theta=theta, tau=tau,  h=torch.sigmoid, n_samples=100,
#     learn_inducing_locations=False, n_inducing_points=50)
#
#
# # In[63]:
#
#
# hist_loss = model.train(
#     epochs=1000,
#     learning_rate=0.1)
#
#
# # In[64]:
#
#
# plt.plot(hist_loss);
#
#
# # In[65]:
#
#
# test_x = torch.linspace(0, 1, 100)
# f_preds = model.pred(test_x)
#
#
# # In[66]:
#
#
# fig, ax = plt.subplots()
# plot_gp_dist(ax, f_preds.numpy(), test_x)
# ax.plot(test_x, u_model(test_x, theta))
# ax.plot(test_x, u_lin(test_x, None), ls=':', color='C2');
#
#
# # ## Run with CPC dataset
#
# # In[67]:
#
#
# df = pd.read_csv("../data/cpc2018.csv")
#
#
# # In[68]:
#
#
# data = df[(df.LotNumB == 1) & (df.LotNumA == 1) & (df.Amb == 0)
#           & (df.Ha >= 0) & (df.Hb >= 0) & (df.La >= 0) & (df.Lb >= 0)]  # & (df.La ==0)  & (df.Lb == 0)
#
#
# # In[69]:
#
#
# data = pd.DataFrame({
#     "subject": data.SubjID,
#     "p0": data.pHa.values,
#     "x0": data.Ha.values,
#     "p1": 1 - data.pHa.values,
#     "x1": data.La.values,
#     "p2": data.pHb.values,
#     "x2": data.Hb.values,
#     "p3": 1 - data.pHb.values,
#     "x3": data.Lb.values,
#     "choices": data.B.values
# })
# max_x = np.max(np.concatenate([data[f'x{i}'] for i in range(4)]))
# for i in range(4):
#     data[f'x{i}'] = data[f'x{i}'] / max_x
#
#
# # In[70]:
#
#
# data
#
#
# # In[71]:
#
#
# opt_param = optimize(data, u=u_model, w=None)
# tau_opt = opt_param[0]
# theta_opt = opt_param[1]
# print("tau", tau_opt, "theta", theta_opt)
#
#
# # In[72]:
#
#
# fig, ax = plt.subplots()
# ax.set_title(f"Population")
# ax.plot(test_x, u_model(test_x, theta_opt))
# plt.show()
#
#
# # ### Try out with one subject
#
# # In[76]:
#
#
# s = 11303
# d = data[data.subject == s]
#
# opt_param = optimize(d, u=u_model, w=None)
# tau = opt_param[0]
# theta = opt_param[1]
# print("tau", tau, "theta", theta)
#
#
# # In[77]:
#
#
# x = np.linspace(-1, 1, 100)
# y = scipy.special.expit(tau_opt*x)
# plt.plot(x, y);
#
#
# # In[78]:
#
#
# u = u_model
#
# model = DiscrepancyModel(
#     data=d,
#     u=u,
#     theta=theta,
#     tau=tau,
#     h=torch.sigmoid,
#     n_samples=100,
#     learn_inducing_locations=False,
#     n_inducing_points=50,
#     jitter=1e-07)
#
# hist_loss = model.train(
#     epochs=1000,
#     learning_rate=0.1)
#
#
# # In[79]:
#
#
# plt.plot(hist_loss)
#
#
# # In[80]:
#
#
# test_x = torch.linspace(0, 1, 100)
# f_preds = model.pred(test_x)
#
# fig, ax = plt.subplots()
#
# ax.set_title(f"Subject {s}; Model = '{u.__name__}'")
# plot_gp_dist(ax, f_preds.numpy(), test_x)
# ax.plot(test_x, u_model(test_x, theta_opt));
#
#
# # In[81]:
#
#
# u = u_lin
#
# model = DiscrepancyModel(
#     data=d,
#     u=u,
#     theta=theta,
#     tau=tau,
#     h=torch.sigmoid,
#     n_samples=100,
#     learn_inducing_locations=False,
#     n_inducing_points=50,
#     jitter=1e-07)
#
# hist_loss = model.train(
#     epochs=1000,
#     learning_rate=0.1)
#
#
# # In[85]:
#
#
# plt.plot(hist_loss);
#
#
# # In[90]:
#
#
# test_x = torch.linspace(0, 1, 100)
# f_preds = model.pred(test_x)
#
# fig, ax = plt.subplots(figsize=(6, 3))
#
# ax.set_title(f"Subject {s}; Model = '{u.__name__}'")
# plot_gp_dist(ax, f_preds.numpy(), test_x)
# ax.plot(test_x, u_model(test_x, theta_opt));
#
#
# # ### Run with all subjects
#
# # In[109]:
#
#
# results = []
#
# counts = data.subject.value_counts()
# subject_325 =  counts[counts == 325].index
#
# for s in tqdm(subject_325):
#
#     try:
#
#         d = data[data.subject == s]
#
#         opt_param = optimize(d, u=u_model, w=None)
#
#         tau = opt_param[0]
#         theta = opt_param[1]
#         print(f"Subject {s} tau={tau} theta={theta}")
#
#         u = u_model
#
#         model = DiscrepancyModel(
#             data=d,
#             u=u,
#             theta=theta,
#             tau=tau,
#             h=torch.sigmoid,
#             n_samples=100,
#             learn_inducing_locations=False,
#             n_inducing_points=50,
#             jitter=1e-07)
#
#         hist_loss = model.train(
#             epochs=1000,
#             learning_rate=0.1)
#
#         test_x = torch.linspace(0, 1, 100)
#         f_preds = model.pred(test_x)
#
#         fig, ax = plt.subplots(figsize=(6, 3))
#
#         ax.set_title(f"Subject {s}; Model = '{u.__name__}'")
#         plot_gp_dist(ax, f_preds.numpy(), test_x)
#         ax.plot(test_x, u_model(test_x, theta))
#
#         plt.show()
#
#         results.append({
#             "subject": s,
#             "discrepancy_model": dill.dumps(model),
#             "hist_loss": hist_loss,
#             "f_preds": f_preds.numpy(),
#             "utility_model": u.__name__,
#             "tau": tau,
#             "theta": theta
#         })
#
#     except Exception as e:
#         print(f"Encountered error with subject {s}: {e}")
#
#
# # In[110]:
#
#
# pd.DataFrame(results).to_csv("results.csv")


# In[ ]:


if __name__ == "__main__":
    run_artificial()
