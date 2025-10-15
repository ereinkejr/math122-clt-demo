import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Central Limit Theorem Demo", layout="wide")

st.title("Central Limit Theorem • Interactive Demo")
st.write(
    "Explore how the distribution of **sample means** approaches normality as sample size increases, "
    "regardless of the population’s shape."
)

# --- Controls
colA, colB, colC = st.columns([1.2,1,1])
with colA:
    dist_name = st.selectbox(
        "Population distribution",
        ["Right-skewed (Exponential)", "Uniform", "Left-skewed (Reverse Exponential)", "Bimodal", "Normal"],
        index=0
    )
with colB:
    n = st.slider("Sample size n", min_value=1, max_value=500, value=30, step=1)
with colC:
    reps = st.slider("Number of samples", min_value=100, max_value=10000, value=3000, step=100)

seed = st.number_input("Random seed (optional)", value=0, min_value=0, step=1)
if seed:
    np.random.seed(seed)

# --- Draw from the chosen population distribution
def sample_population(size):
    if dist_name == "Right-skewed (Exponential)":
        return np.random.exponential(scale=1.0, size=size)
    elif dist_name == "Uniform":
        return np.random.uniform(low=0.0, high=1.0, size=size)
    elif dist_name == "Left-skewed (Reverse Exponential)":
        return 3 - np.random.exponential(scale=1.0, size=size)  # shift to make it mostly on left
    elif dist_name == "Bimodal":
        mix = np.random.rand(size)
        a = np.random.normal(loc=-2, scale=0.7, size=size)
        b = np.random.normal(loc=+2, scale=0.7, size=size)
        return np.where(mix < 0.5, a, b)
    elif dist_name == "Normal":
        return np.random.normal(loc=0.0, scale=1.0, size=size)
    else:
        return np.random.normal(size=size)

# --- Generate reps sample means of size n
pop_preview = sample_population(50_000)
mu_pop = np.mean(pop_preview)
sigma_pop = np.std(pop_preview, ddof=0)

sample_means = []
for _ in range(reps):
    x = sample_population(n)
    sample_means.append(np.mean(x))
sample_means = np.array(sample_means)

mu_means = np.mean(sample_means)
sd_means = np.std(sample_means, ddof=0)
theoretical_sd = sigma_pop / np.sqrt(n) if np.isfinite(sigma_pop) else np.nan

# --- Layout plots
left, right = st.columns(2)

# Population histogram
with left:
    fig1, ax1 = plt.subplots(figsize=(6,4))
    ax1.hist(pop_preview, bins=50, edgecolor="white")
    ax1.set_title("Population distribution (simulated)")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Frequency")
    ax1.axvline(mu_pop, linestyle="--", linewidth=2, label=f"μ ≈ {mu_pop:.3f}")
    ax1.legend()
    st.pyplot(fig1, clear_figure=True)

# Sampling distribution of the mean
with right:
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.hist(sample_means, bins=50, density=True, edgecolor="white")
    ax2.set_title(f"Sampling distribution of the mean (n={n}, reps={reps})")
    ax2.set_xlabel("Sample mean")
    ax2.set_ylabel("Density")
    # Overlay normal with mean mu_pop and sd sigma/√n
    if np.isfinite(theoretical_sd) and theoretical_sd > 0:
        from scipy.stats import norm
        xs = np.linspace(np.min(sample_means), np.max(sample_means), 400)
        ax2.plot(xs, norm.pdf(xs, loc=mu_pop, scale=theoretical_sd), linewidth=2, label="Normal approx (CLT)")
        ax2.legend()
    st.pyplot(fig2, clear_figure=True)

# --- Stats panel
st.subheader("Key results")
c1, c2, c3 = st.columns(3)
c1.metric("Population mean (μ)", f"{mu_pop:.4f}")
c2.metric("Mean of sample means", f"{mu_means:.4f}")
c3.metric("SD of sample means", f"{sd_means:.4f}", help="Empirical SD across simulated sample means")

st.caption(
    f"CLT says: for large n, the distribution of the sample mean is approximately Normal(μ, σ/√n). "
    f"Here σ ≈ {sigma_pop:.4f}, so σ/√n ≈ {theoretical_sd:.4f}."
)

with st.expander("What to look for (teaching tips)"):
    st.markdown(
        """
- With skewed or bimodal populations, the **population histogram** (left) is clearly non-normal.  
- As you increase **n**, the **sampling distribution** (right) tightens and becomes bell-shaped.  
- The center of the sampling distribution stays near the population mean (unbiasedness).  
- The spread roughly matches **σ/√n** (shown by the normal overlay).
        """
    )
