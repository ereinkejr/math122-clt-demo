
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="CLT Visualizer", layout="wide")

st.title("Central Limit Theorem — Interactive Visualizer")

st.write(
    "Explore how the distribution of **sample means** approaches normality as the sample size increases. "
    "Choose any population distribution on the left, set parameters, and compare the population to the sampling distribution."
)

def normal_pdf(x, mu, sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        return np.zeros_like(x, dtype=float)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (np.sqrt(2 * np.pi) * sigma)

def sample_from_distribution(name, size, **params):
    rng = np.random.default_rng(params.get("seed", None))
    if name == "Exponential":
        beta = params.get("beta", 1.0)
        return rng.exponential(beta, size)
    elif name == "Uniform":
        a = params.get("a", 0.0); b = params.get("b", 1.0)
        if a > b: a, b = b, a
        return rng.uniform(a, b, size)
    elif name == "Bernoulli":
        p = params.get("p", 0.3)
        return rng.binomial(1, p, size)
    elif name == "Poisson":
        lam = params.get("lam", 3.0)
        return rng.poisson(lam, size)
    elif name == "Gamma":
        k = params.get("k", 2.0); theta = params.get("theta", 2.0)
        return rng.gamma(shape=k, scale=theta, size=size)
    elif name == "Bimodal":
        w = params.get("w", 0.5)
        mu1, s1 = params.get("mu1", -2.0), params.get("s1", 0.7)
        mu2, s2 = params.get("mu2", 2.0), params.get("s2", 0.7)
        z = rng.random(size)
        x1 = rng.normal(mu1, s1, size)
        x2 = rng.normal(mu2, s2, size)
        return np.where(z < w, x1, x2)
    elif name == "Normal":
        mu = params.get("mu", 0.0); sigma = params.get("sigma", 1.0)
        return rng.normal(mu, sigma, size)
    else:
        return rng.normal(0.0, 1.0, size)

def theory_mu_sigma(name, **params):
    if name == "Exponential":
        beta = params.get("beta", 1.0)
        return beta, beta
    elif name == "Uniform":
        a = params.get("a", 0.0); b = params.get("b", 1.0)
        if a > b: a, b = b, a
        mu = 0.5*(a+b); sigma = (b-a)/np.sqrt(12.0)
        return mu, sigma
    elif name == "Bernoulli":
        p = params.get("p", 0.3)
        return p, np.sqrt(p*(1-p))
    elif name == "Poisson":
        lam = params.get("lam", 3.0)
        return lam, np.sqrt(lam)
    elif name == "Gamma":
        k = params.get("k", 2.0); theta = params.get("theta", 2.0)
        return k*theta, np.sqrt(k)*theta
    elif name == "Normal":
        mu = params.get("mu", 0.0); sigma = params.get("sigma", 1.0)
        return mu, sigma
    elif name == "Bimodal":
        w = params.get("w", 0.5)
        mu1, s1 = params.get("mu1", -2.0), params.get("s1", 0.7)
        mu2, s2 = params.get("mu2", 2.0), params.get("s2", 0.7)
        mu = w*mu1 + (1-w)*mu2
        var = w*(s1**2 + mu1**2) + (1-w)*(s2**2 + mu2**2) - mu**2
        return mu, np.sqrt(max(var, 0.0))
    else:
        return None, None

with st.sidebar:
    st.header("Controls")
    dist_name = st.selectbox(
        "Population distribution",
        ["Exponential", "Uniform", "Bernoulli", "Poisson", "Gamma", "Bimodal", "Normal"],
        index=0
    )

    params = {}
    if dist_name == "Exponential":
        params["beta"] = st.slider("β (mean/scale)", 0.2, 5.0, 1.0, 0.1)
    elif dist_name == "Uniform":
        a = st.slider("a (lower)", -5.0, 5.0, 0.0, 0.1)
        b = st.slider("b (upper)", -5.0, 10.0, 1.0, 0.1)
        params["a"], params["b"] = float(a), float(b)
    elif dist_name == "Bernoulli":
        params["p"] = st.slider("p (success prob.)", 0.01, 0.99, 0.3, 0.01)
    elif dist_name == "Poisson":
        params["lam"] = st.slider("λ (rate/mean)", 0.5, 12.0, 3.0, 0.5)
    elif dist_name == "Gamma":
        params["k"] = st.slider("k (shape)", 0.5, 10.0, 2.0, 0.5)
        params["theta"] = st.slider("θ (scale)", 0.2, 5.0, 2.0, 0.1)
    elif dist_name == "Bimodal":
        params["w"] = st.slider("mixing weight for mode A", 0.0, 1.0, 0.5, 0.05)
        params["mu1"] = st.slider("μ₁", -6.0, 6.0, -2.0, 0.1)
        params["s1"]  = st.slider("σ₁", 0.1, 3.0, 0.7, 0.1)
        params["mu2"] = st.slider("μ₂", -6.0, 6.0,  2.0, 0.1)
        params["s2"]  = st.slider("σ₂", 0.1, 3.0, 0.7, 0.1)
    elif dist_name == "Normal":
        params["mu"] = st.slider("μ", -5.0, 5.0, 0.0, 0.1)
        params["sigma"] = st.slider("σ", 0.1, 4.0, 1.0, 0.1)

    st.markdown("---")
    n = st.slider("Sample size n", 1, 500, 30, 1)
    reps = st.slider("Number of samples (reps)", 200, 20000, 3000, 200)
    bins = st.slider("Histogram bins", 10, 120, 50, 5)
    seed = st.number_input("Random seed (optional)", value=0, min_value=0, step=1)
    if seed:
        params["seed"] = int(seed)

# Population preview for mu/sigma
pop_preview = sample_from_distribution(dist_name, 60000, **params)
mu_pop = float(np.mean(pop_preview))
sigma_pop = float(np.std(pop_preview, ddof=0))

# Sample means
means = np.empty(reps)
for i in range(reps):
    x = sample_from_distribution(dist_name, n, **params)
    means[i] = float(np.mean(x))

mu_means = float(np.mean(means))
sd_means = float(np.std(means, ddof=0))

mu_the, sigma_the = theory_mu_sigma(dist_name, **params)
theoretical_sd = (sigma_the / np.sqrt(n)) if (sigma_the is not None and sigma_the > 0) else np.nan

left, right = st.columns(2, gap="large")

with left:
    fig1, ax1 = plt.subplots(figsize=(6.2,4.2))
    ax1.hist(pop_preview, bins=bins, edgecolor="white")
    ax1.set_title("Population distribution (simulated)", fontsize=12)
    ax1.set_xlabel("Value"); ax1.set_ylabel("Frequency")
    ax1.axvline(mu_pop, linestyle="--", linewidth=2, label=f"μ ≈ {mu_pop:.3f}")
    ax1.legend()
    st.pyplot(fig1, clear_figure=True)

with right:
    fig2, ax2 = plt.subplots(figsize=(6.2,4.2))
    ax2.hist(means, bins=bins, density=True, edgecolor="white")
    ax2.set_title(f"Sampling distribution of the mean (n={n}, reps={reps})", fontsize=12)
    ax2.set_xlabel("Sample mean"); ax2.set_ylabel("Density")

    mu_for_overlay = mu_the if mu_the is not None else mu_pop
    sd_for_overlay = theoretical_sd if np.isfinite(theoretical_sd) and theoretical_sd > 0 else (sigma_pop/np.sqrt(n))
    xs = np.linspace(np.min(means), np.max(means), 400)
    ax2.plot(xs, normal_pdf(xs, mu_for_overlay, sd_for_overlay), linewidth=2, label="Normal approx (CLT)")
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

st.subheader("Key results")
c1, c2, c3 = st.columns(3)
c1.metric("Population mean (μ)", f"{mu_pop:.4f}")
c2.metric("Mean of sample means", f"{mu_means:.4f}")
c3.metric("SD of sample means", f"{sd_means:.4f}")

st.caption(
    f"CLT: for sufficiently large n, \bar X ≈ Normal(μ, σ/√n). "
    f"Here, σ (population SD) ≈ {sigma_pop:.4f} → σ/√n ≈ {sd_for_overlay:.4f}. "
    "Even for skewed or bimodal populations, the sampling distribution tends toward normality as n grows."
)

with st.expander("Teaching prompts"):
    st.markdown(
        """
- Start with a **skewed** or **bimodal** population: increase **n** and watch the sampling distribution become bell-shaped.
- Compare **empirical SD of sample means** to the theoretical value **σ/√n** by adjusting **n**.
- Ask: for which distributions does the CLT approximation seem reasonable at smaller **n**?
- Highlight that the **center** of the sampling distribution stays near the population mean (unbiasedness).
        """
    )
