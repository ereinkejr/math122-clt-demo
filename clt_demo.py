
import numpy as np
import matplotlib.pyplot as plt

def _rng_from_source(source, size, **kwargs):
    rng = np.random.default_rng(kwargs.get("seed", None))
    if source == "uniform":
        a = kwargs.get("a", 0.0)
        b = kwargs.get("b", 1.0)
        return rng.uniform(a, b, size)
    elif source == "exponential":
        beta = kwargs.get("beta", 1.0)
        return rng.exponential(beta, size)
    elif source == "bernoulli":
        p = kwargs.get("p", 0.3)
        return rng.binomial(1, p, size)
    elif source == "poisson":
        lam = kwargs.get("lam", 3.0)
        return rng.poisson(lam, size)
    elif source == "gamma":
        k = kwargs.get("k", 2.0)
        theta = kwargs.get("theta", 2.0)
        return rng.gamma(k, theta, size)
    elif source == "bimodal":
        w = kwargs.get("w", 0.6)
        mu1, s1 = kwargs.get("mu1", 0.0), kwargs.get("s1", 1.0)
        mu2, s2 = kwargs.get("mu2", 6.0), kwargs.get("s2", 1.5)
        z = rng.random(size)
        x1 = rng.normal(mu1, s1, size)
        x2 = rng.normal(mu2, s2, size)
        return np.where(z < w, x1, x2)
    else:
        raise ValueError(f"Unknown source '{source}'.")

def _theory_mean_std(source, **kwargs):
    if source == "uniform":
        a = kwargs.get("a", 0.0); b = kwargs.get("b", 1.0)
        mu = 0.5 * (a + b)
        sigma = (b - a) / np.sqrt(12.0)
        return mu, sigma
    elif source == "exponential":
        beta = kwargs.get("beta", 1.0)
        mu = beta
        sigma = beta
        return mu, sigma
    elif source == "bernoulli":
        p = kwargs.get("p", 0.3)
        mu = p
        sigma = np.sqrt(p * (1 - p))
        return mu, sigma
    elif source == "poisson":
        lam = kwargs.get("lam", 3.0)
        mu = lam
        sigma = np.sqrt(lam)
        return mu, sigma
    elif source == "gamma":
        k = kwargs.get("k", 2.0); theta = kwargs.get("theta", 2.0)
        mu = k * theta
        sigma = np.sqrt(k) * theta
        return mu, sigma
    elif source == "bimodal":
        w = kwargs.get("w", 0.6)
        mu1, s1 = kwargs.get("mu1", 0.0), kwargs.get("s1", 1.0)
        mu2, s2 = kwargs.get("mu2", 6.0), kwargs.get("s2", 1.5)
        mu = w * mu1 + (1 - w) * mu2
        var = w * (s1**2 + mu1**2) + (1 - w) * (s2**2 + mu2**2) - mu**2
        sigma = np.sqrt(var)
        return mu, sigma
    else:
        return None, None

def simulate_sample_means(source="exponential", n=5, reps=2000, **kwargs):
    rng = np.random.default_rng(kwargs.get("seed", None))
    samples = _rng_from_source(source, size=(reps, n), **kwargs)
    return samples.mean(axis=1)

def clt_grid(source="exponential", n_list=(1,2,5,10,30,50), reps=4000, bins=40, figsize=(12,8), sharex=True, **kwargs):
    mu, sigma = _theory_mean_std(source, **kwargs)
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=sharex, sharey=True)
    axes = axes.ravel()

    if mu is not None and sigma is not None:
        x_min = mu - 4 * sigma / np.sqrt(max(n_list))
        x_max = mu + 4 * sigma / np.sqrt(min(n_list))
    else:
        pilot = _rng_from_source(source, size=20000, **kwargs)
        mu = pilot.mean() if mu is None else mu
        sigma = pilot.std(ddof=0) if sigma is None else sigma
        x_min = mu - 4 * sigma / np.sqrt(max(n_list))
        x_max = mu + 4 * sigma / np.sqrt(min(n_list))

    for ax, n in zip(axes, n_list):
        means = simulate_sample_means(source=source, n=n, reps=reps, **kwargs)
        ax.hist(means, bins=bins, density=True, alpha=0.8, edgecolor="white")
        xs = np.linspace(x_min, x_max, 501)
        ax.plot(xs, 1/np.sqrt(2*np.pi*(sigma**2/n)) * np.exp(-(xs-mu)**2/(2*(sigma**2/n))), lw=2)
        ax.set_title(f"n = {n}")
        ax.set_xlabel("sample mean")
        ax.set_ylabel("density")
        emp_mu = means.mean()
        emp_sd = means.std(ddof=1)
        ax.text(0.02, 0.95, f"emp μ={emp_mu:.3g}\nemp σ={emp_sd:.3g}\nθ σ={sigma/np.sqrt(n):.3g}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9)

    fig.suptitle(f"Central Limit Theorem demo — source: {source}", y=0.995, fontsize=14)
    fig.tight_layout()
    return fig

def clt_interactive():
    import ipywidgets as W
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt

    source_dd = W.Dropdown(
        options=[("Exponential", "exponential"),
                 ("Uniform(0,1)", "uniform"),
                 ("Bernoulli(p=0.3)", "bernoulli"),
                 ("Poisson(λ=3)", "poisson"),
                 ("Gamma(k=2,θ=2)", "gamma"),
                 ("Bimodal (two normals)", "bimodal")],
        value="exponential",
        description="Source:",
        layout=W.Layout(width="300px")
    )
    reps_slider = W.IntSlider(value=4000, min=500, max=20000, step=500, description="reps")
    bins_slider = W.IntSlider(value=40, min=10, max=100, step=5, description="bins")
    n_checks = W.SelectMultiple(options=[1,2,5,10,30,50,100], value=(1,2,5,10,30,50), description="n_list", rows=7)
    seed_box = W.IntText(value=0, description="seed", layout=W.Layout(width="160px"))

    p_box = W.FloatSlider(value=0.3, min=0.05, max=0.95, step=0.05, description="p (Bernoulli)")
    lam_box = W.FloatSlider(value=3.0, min=0.5, max=10.0, step=0.5, description="λ (Poisson)")
    beta_box = W.FloatSlider(value=1.0, min=0.2, max=5.0, step=0.1, description="β (Expo)")
    a_box = W.FloatSlider(value=0.0, min=-5.0, max=5.0, step=0.1, description="a (Unif)")
    b_box = W.FloatSlider(value=1.0, min=-5.0, max=10.0, step=0.1, description="b (Unif)")
    k_box = W.FloatSlider(value=2.0, min=0.5, max=10.0, step=0.5, description="k (Gamma)")
    theta_box = W.FloatSlider(value=2.0, min=0.2, max=5.0, step=0.1, description="θ (Gamma)")

    out = W.Output()

    def render(*_):
        with out:
            clear_output(wait=True)
            kwargs = {"seed": seed_box.value}
            src = source_dd.value
            if src == "bernoulli":
                kwargs["p"] = p_box.value
            elif src == "poisson":
                kwargs["lam"] = lam_box.value
            elif src == "exponential":
                kwargs["beta"] = beta_box.value
            elif src == "uniform":
                a = min(a_box.value, b_box.value)
                b = max(a_box.value, b_box.value)
                kwargs["a"] = a; kwargs["b"] = b
            elif src == "gamma":
                kwargs["k"] = k_box.value; kwargs["theta"] = theta_box.value

            fig = clt_grid(
                source=src,
                n_list=tuple(n_checks.value),
                reps=reps_slider.value,
                bins=bins_slider.value,
                **kwargs
            )
            display(fig)
            plt.close(fig)

    for w in [source_dd, reps_slider, bins_slider, n_checks, seed_box, p_box, lam_box, beta_box, a_box, b_box, k_box, theta_box]:
        w.observe(render, names="value")

    controls_left = W.VBox([source_dd, n_checks, reps_slider, bins_slider, seed_box])
    controls_right = W.VBox([p_box, lam_box, beta_box, a_box, b_box, k_box, theta_box])
    ui = W.HBox([controls_left, controls_right])
    display(ui, out)
    render()
