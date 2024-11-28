import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings


# Geometric distribution
def geometric(p, vals):
    if not (0 < p <= 1):
        raise ValueError("Parameter 'p' for geometric distribution must be in the range (0, 1].")
    return stats.geom.pmf(vals, p)


# Binomial distribution
def binomial(n, p, vals):
    if n <= 0 or not isinstance(n, int):
        raise ValueError("Parameter 'n' for binomial distribution must be a positive integer.")
    if not (0 <= p <= 1):
        raise ValueError("Parameter 'p' for binomial distribution must be in the range [0, 1].")
    return stats.binom.pmf(vals, n, p)


# Poisson distribution
def poisson(lam, vals):
    if lam <= 0:
        raise ValueError("Parameter 'lam' (lambda) for Poisson distribution must be a positive number.")
    return stats.poisson.pmf(vals, lam)


# Uniform distribution
def uniform(a, b, vals):
    if a >= b:
        raise ValueError("Parameter 'a' must be less than 'b' for uniform distribution.")
    return stats.uniform.pdf(vals, loc=a, scale=b - a)


# Main function to handle distributions
def get_probability_distributions(dist_name, params, vals):
    if not isinstance(vals, list):
        raise ValueError("Parameter 'vals' must be a list.")
    if any(v < 0 for v in vals):
        warnings.warn("Negative values in 'vals' are removed.")
        vals = [v for v in vals if v >= 0]
        if not vals:
            raise ValueError("All numbers in 'vals' are negative. Provide non-negative values.")

    vals = np.array(vals)

    if dist_name.lower() == "geometric":
        if "p" not in params:
            raise ValueError("Missing parameter 'p' for geometric distribution.")
        p = params["p"]
        probabilities = geometric(p, vals)

    elif dist_name.lower() == "binomial":
        if "n" not in params or "p" not in params:
            raise ValueError("Missing parameters 'n' and/or 'p' for binomial distribution.")
        n = params["n"]
        p = params["p"]
        probabilities = binomial(n, p, vals)

    elif dist_name.lower() == "poisson":
        if "lam" not in params:
            raise ValueError("Missing parameter 'lam' for Poisson distribution.")
        lam = params["lam"]
        probabilities = poisson(lam, vals)

    elif dist_name.lower() == "uniform":
        if "a" not in params or "b" not in params:
            raise ValueError("Missing parameters 'a' and/or 'b' for uniform distribution.")
        a = params["a"]
        b = params["b"]
        probabilities = uniform(a, b, vals)

    else:
        raise ValueError(f"Unknown distribution name: {dist_name}")

    # Plot the distribution
    plt.bar(vals, probabilities, alpha=0.7, label=dist_name.capitalize())
    plt.xlabel("Values")
    plt.ylabel("Probability")
    plt.title(f"{dist_name.capitalize()} Distribution")
    plt.legend()
    plt.show()

    return probabilities


# Example Usage
try:
    dist_name = "binomial"
    params = {"n": 10, "p": 0.5}
    vals = [0, 1, 2, 3, 4, 5, -1, -2]
    probabilities = get_probability_distributions(dist_name, params, vals)
    print("Probabilities:", probabilities)
except Exception as e:
    print("Error:", e)
