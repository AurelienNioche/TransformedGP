from torch.distributions import transform_to


def rescale(x):
    x = (x - x.mean()) / x.std()
    return x


class Jpps(nn.Module):

    def __init__(self):
        super().__init__()
        self.unc_param = nn.Parameter(torch.ones(8))

    def forward(self, x):
        param = torch.nn.functional.softplus(self.unc_param)

        tp = x + 0.75

        pred = param[0] \
               * (1 - 1 / (1 + (tp / param[1]) ** param[2]
                           + (tp / param[3]) ** param[4] + (tp / param[5]) **
                           param[6]))
        return torch.distributions.Normal(pred, param[7])


epochs = 5000
init_lr = 0.5

# loss = nn.MSELoss()
model = Jpps()

x = torch.from_numpy(data.age.values.astype(np.float32))
y = torch.from_numpy(data.height.values.astype(np.float32))

optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

# with torch.no_grad():
#     previous_param = model.unc_param[:].clone()

hist_loss = []

for epoch in range(epochs):
    # Zero gradients from previous iteration
    optimizer.zero_grad()

    output = model(x)
    loss = - output.log_prob(y).sum()
    # output = loss(input, target)
    loss.backward()
    optimizer.step()

    hist_loss.append(loss.item())

#     with torch.no_grad():
#         #if torch.isclose(previous_param, model.unc_param).all():
#         #    print(f"epoch {epoch} converged!")
#         #    break

#         previous_param = model.unc_param[:].clone()

with torch.no_grad():
    model.eval()

    print(model.unc_param)

    x_plot = np.linspace(0, 20, 100)
    pred_dist = model.forward(x_plot)
    pred_loc = pred_dist.loc

    lower = pred_loc - 2 * pred_dist.scale
    upper = pred_loc + 2 * pred_dist.scale

    fig, ax = plt.subplots()
    ax.scatter(data.age, data.height, alpha=0.2)
    ax.set_title("JPPS")
    ax.plot(x_plot, pred_loc.numpy(), color="C1")
    ax.fill_between(x_plot, lower.numpy(), upper.numpy(), alpha=0.3)
    ax.set_xlabel("age")
    ax.set_ylabel("height");