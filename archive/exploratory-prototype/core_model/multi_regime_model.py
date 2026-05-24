import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.optimize import curve_fit

# Set random seed for reproducibility
np.random.seed(42)

# =====================
# Model Definitions
# =====================

def multi_regime_model(params, h, c, e):
    """
    Multi-regime model that doesn't assume specific curve shape
    
    E(h, c, e) = β₁ + β₂(h)^α₁ + β₃(c)^α₂ + β₄(e)^α₃ + β₅(h×c×e)^α₄
    
    Args:
        params: List of parameters [β₁, β₂, β₃, β₄, β₅, α₁, α₂, α₃, α₄]
        h: Harm potential (1-10)
        c: Complexity (1-10)
        e: Error rate (0.01-0.50)
        
    Returns:
        Required expertise level (1-10)
    """
    β1, β2, β3, β4, β5, α1, α2, α3, α4 = params
    
    # Handle potential numerical issues with powers
    term1 = β1
    term2 = β2 * np.power(h, α1) if h > 0 else 0
    term3 = β3 * np.power(c, α2) if c > 0 else 0
    term4 = β4 * np.power(e, α3) if e > 0 else 0
    term5 = β5 * np.power(h * c * e, α4) if h * c * e > 0 else 0
    
    expertise = term1 + term2 + term3 + term4 + term5
    
    # Cap expertise between 1 and 10
    return np.clip(expertise, 1, 10)

def linear_model(params, h, c, e):
    """
    Simple linear model
    
    E(h, c, e) = b0 + b1*h + b2*c + b3*e
    """
    b0, b1, b2, b3 = params
    expertise = b0 + b1*h + b2*c + b3*e
    return np.clip(expertise, 1, 10)

def sigmoid_model(params, h, c, e):
    """
    Sigmoid model (S-curve)
    
    E(h) = E_min + (E_max - E_min) * (1 / (1 + np.exp(-k * (h - h0))))
    """
    E_min, E_max, k, h0 = params
    expertise = E_min + (E_max - E_min) * (1 / (1 + np.exp(-k * (h - h0))))
    return np.clip(expertise, 1, 10)

def power_model(params, h, c, e):
    """
    Power law model
    
    E(h, c, e) = a * h^b
    """
    a, b = params
    expertise = a * np.power(h, b)
    return np.clip(expertise, 1, 10)

def threshold_model(params, h, c, e):
    """
    Threshold model with different slopes before and after threshold
    
    E(h) = a1*h if h < threshold else a2*h + b
    """
    a1, a2, threshold, b = params
    expertise = np.where(h < threshold, a1 * h, a2 * h + b)
    return np.clip(expertise, 1, 10)

# Dictionary of models for easier reference
MODELS = {
    "Multi-Regime": {
        "function": multi_regime_model,
        "initial_params": [1, 0.1, 0.1, 0.1, 0.01, 1, 1, 1, 1],  # [β₁, β₂, β₃, β₄, β₅, α₁, α₂, α₃, α₄]
        "param_bounds": [(0, 5), (0, 3), (0, 3), (0, 3), (0, 1), (0.5, 3), (0.5, 3), (0.5, 3), (0.5, 3)]
    },
    "Linear": {
        "function": linear_model,
        "initial_params": [1, 0.5, 0.2, 0.1],  # [b0, b1, b2, b3]
        "param_bounds": [(0, 5), (0, 3), (0, 3), (0, 3)]
    },
    "Sigmoid": {
        "function": sigmoid_model,
        "initial_params": [1, 10, 1, 5],  # [E_min, E_max, k, h0]
        "param_bounds": [(1, 3), (8, 10), (0.5, 3), (3, 7)]
    },
    "Power": {
        "function": power_model,
        "initial_params": [1, 1],  # [a, b]
        "param_bounds": [(0.1, 3), (0.5, 3)]
    },
    "Threshold": {
        "function": threshold_model,
        "initial_params": [0.5, 1.5, 5, -2.5],  # [a1, a2, threshold, b]
        "param_bounds": [(0.1, 1), (1, 3), (3, 7), (-5, 0)]
    }
}

# =====================
# Domain Definitions
# =====================

# Define domains and their empirically-derived characteristics
DOMAINS = {
    "Medical": {
        "error_rate_min": 0.01,   # Minimum error rate from research
        "error_rate_max": 0.246,  # Maximum error rate from research
        "error_rate_mode": 0.15,  # Mode/most common error rate (estimated)
        "stakes": 9,              # High stakes (1-10)
        "complexity": 8,          # High complexity (1-10)
        "severity_alpha": 1.2,    # Power law param - higher alpha = more severe errors
        "expert_mitigation": 0.7  # Experts can mitigate 70% of harm
    },
    "Legal": {
        "error_rate_min": 0.05,   # Minimum error rate for best models
        "error_rate_max": 0.88,   # Maximum error rate from research
        "error_rate_mode": 0.69,  # Mode error rate (from GPT-3.5 study)
        "stakes": 8,              # High stakes (1-10)
        "complexity": 7,          # High complexity (1-10)
        "severity_alpha": 1.5,    # Severe errors but not as severe as medical
        "expert_mitigation": 0.65 # Experts can mitigate 65% of harm
    },
    "Software": {
        "error_rate_min": 0.01,   # For best models
        "error_rate_max": 0.217,  # For open source models 
        "error_rate_mode": 0.1,   # Estimated mode
        "stakes": 6,              # Medium-high stakes (1-10)
        "complexity": 6,          # Medium-high complexity (1-10)
        "severity_alpha": 1.8,    # Less severe errors on average
        "expert_mitigation": 0.8  # Experts can mitigate 80% of harm (easier to test)
    },
    "Academic": {
        "error_rate_min": 0.01,   # For best models
        "error_rate_max": 0.91,   # Bard/Gemini citation hallucination rate
        "error_rate_mode": 0.47,  # Mode from ChatGPT-3.5 medical citation study
        "stakes": 7,              # Medium-high stakes (1-10)
        "complexity": 7,          # High complexity (1-10)
        "severity_alpha": 1.4,    # Higher severity - academic integrity issues
        "expert_mitigation": 0.75 # Experts can mitigate 75% of harm
    },
    "Creative": {
        "error_rate_min": 0.01,   # Low error rate for creative content
        "error_rate_max": 0.15,   # Maximum error rate (estimated)
        "error_rate_mode": 0.05,  # Mode error rate (estimated)
        "stakes": 3,              # Lower stakes (1-10)
        "complexity": 4,          # Medium complexity (1-10)
        "severity_alpha": 2.5,    # Mostly minor errors
        "expert_mitigation": 0.9  # Experts can mitigate 90% of harm
    }
}

# Expertise level definitions
EXPERTISE_LEVELS = {
    "Novice": (1, 3),
    "Familiar": (3, 6),
    "Educated": (6, 9),
    "Expert": (9, 10)
}

# =====================
# Utility Functions
# =====================

def sample_error_rate(domain_params):
    """
    Sample error rate using a triangular distribution based on domain parameters.
    This better reflects the empirical data than uniform distribution.
    """
    return np.random.triangular(
        domain_params["error_rate_min"],
        domain_params["error_rate_mode"],
        domain_params["error_rate_max"]
    )

def sample_error_severity(alpha, size=1):
    """
    Sample error severity using a power law distribution.
    Alpha controls shape - higher alpha = more minor errors, fewer severe ones.
    Output scaled to 1-10 range.
    """
    # Generate a power law distribution
    raw_severity = np.random.pareto(alpha, size=size)
    
    # Scale to 1-10 range (capped)
    severity = 1 + 9 * np.minimum(raw_severity / np.max([5, np.percentile(raw_severity, 99)]), 1)
    
    return severity if size > 1 else severity[0]

def get_expertise_category(expertise_level):
    """Convert numerical expertise level to category."""
    for category, (min_val, max_val) in EXPERTISE_LEVELS.items():
        if min_val <= expertise_level <= max_val:
            return category
    return "Unknown"

# =====================
# Simulation Functions
# =====================

def generate_simulation_data(num_samples=5000):
    """
    Generate simulation data across all domains.
    
    Args:
        num_samples: Number of samples per domain
        
    Returns:
        pandas.DataFrame: Simulation data
    """
    data = []
    
    for domain, params in tqdm(DOMAINS.items(), desc="Generating domain data"):
        for _ in range(num_samples):
            # Sample parameters
            harm = np.random.uniform(1, 10)
            error_rate = sample_error_rate(params)
            error_severity = sample_error_severity(params["severity_alpha"])
            complexity = params["complexity"]
            
            # Store data
            data.append({
                "Domain": domain,
                "Harm": harm,
                "Complexity": complexity,
                "Error_Rate": error_rate,
                "Error_Severity": error_severity,
                "Stakes": params["stakes"],
                "Severity_Alpha": params["severity_alpha"],
                "Expert_Mitigation": params["expert_mitigation"]
            })
    
    return pd.DataFrame(data)

def fit_model(model_name, X, y):
    """
    Fit a specific model to the data.
    
    Args:
        model_name: Name of the model to fit
        X: Input data (harm, complexity, error_rate)
        y: Target data (expertise)
        
    Returns:
        tuple: (optimal parameters, R-squared)
    """
    model_info = MODELS[model_name]
    model_func = model_info["function"]
    initial_params = model_info["initial_params"]
    param_bounds = model_info["param_bounds"]
    
    try:
        # Define wrapper function for curve_fit
        def fit_func(X, *params):
            h, c, e = X.T
            return model_func(params, h, c, e)
        
        # Fit the model
        popt, _ = curve_fit(
            fit_func, 
            X, 
            y, 
            p0=initial_params,
            bounds=tuple(zip(*param_bounds)),
            maxfev=10000
        )
        
        # Calculate predictions
        y_pred = model_func(popt, *X.T)
        
        # Calculate R-squared
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        
        return popt, r_squared
    
    except Exception as e:
        print(f"Error fitting {model_name} model: {e}")
        return initial_params, 0.0

def generate_expertise_data(simulation_data, domain_specific_models=True):
    """
    Generate expertise data by fitting models to simulation data.
    
    Args:
        simulation_data: DataFrame with simulation data
        domain_specific_models: Whether to fit models for each domain separately
        
    Returns:
        tuple: (DataFrame with expertise data, dict with fitted models)
    """
    expertise_data = simulation_data.copy()
    fitted_models = {}
    
    if domain_specific_models:
        # Fit models for each domain separately
        domains = expertise_data["Domain"].unique()
        
        for domain in tqdm(domains, desc="Fitting domain models"):
            domain_data = expertise_data[expertise_data["Domain"] == domain]
            
            # Assume true expertise is based on multi-regime model with domain-specific parameters
            # These parameters would be determined by domain experts in a real application
            # For simulation, we'll generate realistic parameters
            
            # Generate domain-specific parameters for multi-regime model
            np.random.seed(hash(domain) % 10000)  # Domain-specific seed
            true_params = [
                1.0,  # β₁: Base expertise
                0.05 + 0.3 * domain_data["Stakes"].iloc[0] / 10,  # β₂: Harm sensitivity
                0.05 + 0.2 * domain_data["Complexity"].iloc[0] / 10,  # β₃: Complexity sensitivity
                0.05 + 0.1 * (1 - domain_data["Expert_Mitigation"].iloc[0]),  # β₄: Error sensitivity
                0.01 + 0.02 * domain_data["Stakes"].iloc[0] / 10,  # β₅: Interaction sensitivity
                0.8 + 0.6 * domain_data["Stakes"].iloc[0] / 10,  # α₁: Harm exponent
                0.8 + 0.6 * domain_data["Complexity"].iloc[0] / 10,  # α₂: Complexity exponent
                0.8 + 0.2 * (1 - domain_data["Expert_Mitigation"].iloc[0]),  # α₃: Error exponent
                0.5 + 1.5 * domain_data["Severity_Alpha"].iloc[0] / 3  # α₄: Interaction exponent
            ]
            
            # Calculate true expertise
            X_domain = domain_data[["Harm", "Complexity", "Error_Rate"]].values
            domain_data["True_Expertise"] = multi_regime_model(true_params, *X_domain.T)
            
            # Fit all models to the data
            fitted_models[domain] = {}
            for model_name in MODELS.keys():
                params, r2 = fit_model(model_name, X_domain, domain_data["True_Expertise"].values)
                fitted_models[domain][model_name] = {"params": params, "r2": r2}
            
            # Generate predictions for all models
            for model_name, model_info in MODELS.items():
                params = fitted_models[domain][model_name]["params"]
                domain_data[f"{model_name}_Expertise"] = model_info["function"](params, *X_domain.T)
            
            # Update expertise data
            expertise_data.loc[expertise_data["Domain"] == domain] = domain_data
    
    else:
        # Fit a single model across all domains
        # (This is less accurate but useful for comparison)
        X_all = expertise_data[["Harm", "Complexity", "Error_Rate"]].values
        
        # Generate a generic set of "true" parameters
        np.random.seed(0)
        true_params = [1.0, 0.2, 0.15, 0.1, 0.01, 1.2, 1.1, 0.9, 1.0]
        
        # Calculate true expertise
        expertise_data["True_Expertise"] = multi_regime_model(true_params, *X_all.T)
        
        # Fit all models to the data
        fitted_models["all"] = {}
        for model_name in MODELS.keys():
            params, r2 = fit_model(model_name, X_all, expertise_data["True_Expertise"].values)
            fitted_models["all"][model_name] = {"params": params, "r2": r2}
        
        # Generate predictions for all models
        for model_name, model_info in MODELS.items():
            params = fitted_models["all"][model_name]["params"]
            expertise_data[f"{model_name}_Expertise"] = model_info["function"](params, *X_all.T)
    
    return expertise_data, fitted_models

def calculate_model_errors(expertise_data, model_names=None):
    """
    Calculate model errors for each model.
    
    Args:
        expertise_data: DataFrame with expertise data
        model_names: List of model names to evaluate
        
    Returns:
        pandas.DataFrame: Error metrics for each model
    """
    if model_names is None:
        model_names = [name for name in MODELS.keys()]
    
    error_metrics = []
    
    for domain in expertise_data["Domain"].unique():
        domain_data = expertise_data[expertise_data["Domain"] == domain]
        
        for model_name in model_names:
            # Calculate error metrics
            y_true = domain_data["True_Expertise"]
            y_pred = domain_data[f"{model_name}_Expertise"]
            
            mse = np.mean((y_true - y_pred) ** 2)
            mae = np.mean(np.abs(y_true - y_pred))
            max_error = np.max(np.abs(y_true - y_pred))
            
            error_metrics.append({
                "Domain": domain,
                "Model": model_name,
                "MSE": mse,
                "MAE": mae,
                "Max_Error": max_error
            })
    
    return pd.DataFrame(error_metrics)

# =====================
# Visualization Functions
# =====================

def plot_model_comparison(expertise_data, domain=None):
    """
    Plot comparison of different models for a given domain.
    
    Args:
        expertise_data: DataFrame with expertise data
        domain: Domain to plot (if None, plots for all domains)
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    if domain is not None:
        domains = [domain]
    else:
        domains = expertise_data["Domain"].unique()
    
    fig, axes = plt.subplots(len(domains), 1, figsize=(12, 6 * len(domains)))
    if len(domains) == 1:
        axes = [axes]
    
    for i, domain in enumerate(domains):
        ax = axes[i]
        domain_data = expertise_data[expertise_data["Domain"] == domain]
        
        # Sort by harm for better visualization
        domain_data = domain_data.sort_values("Harm")
        
        # Plot true expertise
        ax.plot(domain_data["Harm"], domain_data["True_Expertise"], 'k-', label="True Expertise", linewidth=3)
        
        # Plot model predictions
        for model_name in MODELS.keys():
            ax.plot(domain_data["Harm"], domain_data[f"{model_name}_Expertise"], '--', label=f"{model_name} Model")
        
        ax.set_title(f"{domain} Domain: Model Comparison", fontsize=16)
        ax.set_xlabel("Harm Potential (1-10)", fontsize=14)
        ax.set_ylabel("Required Expertise (1-10)", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    plt.tight_layout()
    return fig

def plot_model_errors(error_metrics):
    """
    Plot model errors for each domain.
    
    Args:
        error_metrics: DataFrame with error metrics
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot MSE
    sns.barplot(x="Domain", y="MSE", hue="Model", data=error_metrics, ax=axes[0])
    axes[0].set_title("Mean Squared Error by Domain", fontsize=16)
    axes[0].set_ylabel("MSE", fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot MAE
    sns.barplot(x="Domain", y="MAE", hue="Model", data=error_metrics, ax=axes[1])
    axes[1].set_title("Mean Absolute Error by Domain", fontsize=16)
    axes[1].set_ylabel("MAE", fontsize=14)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot Max Error
    sns.barplot(x="Domain", y="Max_Error", hue="Model", data=error_metrics, ax=axes[2])
    axes[2].set_title("Maximum Error by Domain", fontsize=16)
    axes[2].set_ylabel("Max Error", fontsize=14)
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def plot_expertise_heatmap(expertise_data, domain):
    """
    Plot heatmap of expertise vs. harm and error rate for a given domain.
    
    Args:
        expertise_data: DataFrame with expertise data
        domain: Domain to plot
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    domain_data = expertise_data[expertise_data["Domain"] == domain]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Create a pivot table for true expertise
    heatmap_data = domain_data.pivot_table(
        values="True_Expertise",
        index=pd.cut(domain_data["Error_Rate"], bins=10),
        columns=pd.cut(domain_data["Harm"], bins=10),
        aggfunc="mean"
    )
    
    # Plot heatmap for true expertise
    sns.heatmap(heatmap_data, cmap="viridis", ax=axes[0], annot=True, fmt=".1f")
    axes[0].set_title(f"{domain} Domain: True Expertise by Harm and Error Rate", fontsize=16)
    axes[0].set_ylabel("Error Rate", fontsize=14)
    axes[0].set_xlabel("Harm Potential", fontsize=14)
    
    # Find best fitting model for this domain
    model_performance = error_metrics[error_metrics["Domain"] == domain]
    best_model = model_performance.loc[model_performance["MSE"].idxmin(), "Model"]
    
    # Create a pivot table for best model expertise
    heatmap_data = domain_data.pivot_table(
        values=f"{best_model}_Expertise",
        index=pd.cut(domain_data["Error_Rate"], bins=10),
        columns=pd.cut(domain_data["Harm"], bins=10),
        aggfunc="mean"
    )
    
    # Plot heatmap for best model expertise
    sns.heatmap(heatmap_data, cmap="viridis", ax=axes[1], annot=True, fmt=".1f")
    axes[1].set_title(f"{domain} Domain: {best_model} Model Expertise by Harm and Error Rate", fontsize=16)
    axes[1].set_ylabel("Error Rate", fontsize=14)
    axes[1].set_xlabel("Harm Potential", fontsize=14)
    
    plt.tight_layout()
    return fig

def plot_domain_comparison(expertise_data):
    """
    Plot comparison of expertise profiles across domains.
    
    Args:
        expertise_data: DataFrame with expertise data
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Group by domain and calculate mean expertise
    domain_expertise = expertise_data.groupby("Domain")["True_Expertise"].mean().sort_values(ascending=False)
    
    # Plot mean expertise by domain
    sns.barplot(x=domain_expertise.index, y=domain_expertise.values, ax=axes[0])
    axes[0].set_title("Mean Required Expertise by Domain", fontsize=16)
    axes[0].set_ylabel("Mean Expertise (1-10)", fontsize=14)
    axes[0].set_xlabel("Domain", fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot expertise distribution by domain
    sns.boxplot(x="Domain", y="True_Expertise", data=expertise_data, ax=axes[1])
    axes[1].set_title("Expertise Distribution by Domain", fontsize=16)
    axes[1].set_ylabel("Required Expertise (1-10)", fontsize=14)
    axes[1].set_xlabel("Domain", fontsize=14)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def plot_parameter_correlations(expertise_data):
    """
    Plot correlations between parameters and expertise.
    
    Args:
        expertise_data: DataFrame with expertise data
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, axes = plt.subplots(len(DOMAINS), 1, figsize=(12, 5 * len(DOMAINS)))
    
    for i, domain in enumerate(DOMAINS.keys()):
        ax = axes[i]
        domain_data = expertise_data[expertise_data["Domain"] == domain]
        
        # Calculate correlations
        corr_data = domain_data[["Harm", "Complexity", "Error_Rate", "Error_Severity", "True_Expertise"]].corr()["True_Expertise"].drop("True_Expertise")
        
        # Plot correlations
        sns.barplot(x=corr_data.index, y=corr_data.values, ax=ax)
        ax.set_title(f"{domain} Domain: Parameter Correlations with Expertise", fontsize=16)
        ax.set_ylabel("Correlation Coefficient", fontsize=14)
        ax.set_ylim(-1, 1)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

# =====================
# Main Function
# =====================

def main():
    print("Generating simulation data...")
    simulation_data = generate_simulation_data(num_samples=2000)
    
    print("Fitting models and generating expertise data...")
    expertise_data, fitted_models = generate_expertise_data(simulation_data)
    
    print("Calculating model errors...")
    global error_metrics
    error_metrics = calculate_model_errors(expertise_data)
    
    # Print model performance summary
    print("\n=== Model Performance Summary ===")
    for domain in expertise_data["Domain"].unique():
        print(f"\n{domain} Domain:")
        domain_metrics = error_metrics[error_metrics["Domain"] == domain]
        for model_name in MODELS.keys():
            model_metric = domain_metrics[domain_metrics["Model"] == model_name].iloc[0]
            print(f"  {model_name} Model: MSE = {model_metric['MSE']:.4f}, MAE = {model_metric['MAE']:.4f}")
        
        # Print best model for this domain
        best_model = domain_metrics.loc[domain_metrics["MSE"].idxmin(), "Model"]
        print(f"  Best Model: {best_model}")
        
        # Print fitted parameters for multi-regime model
        if domain in fitted_models:
            multi_params = fitted_models[domain]["Multi-Regime"]["params"]
            print(f"  Multi-Regime Parameters: β = {multi_params[:5]}, α = {multi_params[5:]}")
    
    # Create and save visualizations
    print("\nCreating visualizations...")
    
    # Plot model comparison for each domain
    for domain in expertise_data["Domain"].unique():
        fig = plot_model_comparison(expertise_data, domain)
        fig.savefig(f"{domain}_model_comparison.png")
        plt.close(fig)
    
    # Plot model errors
    fig = plot_model_errors(error_metrics)
    fig.savefig("model_errors.png")
    plt.close(fig)
    
    # Plot expertise heatmap for each domain
    for domain in expertise_data["Domain"].unique():
        fig = plot_expertise_heatmap(expertise_data, domain)
        fig.savefig(f"{domain}_expertise_heatmap.png")
        plt.close(fig)
    
    # Plot domain comparison
    fig = plot_domain_comparison(expertise_data)
    fig.savefig("domain_comparison.png")
    plt.close(fig)
    
    # Plot parameter correlations
    fig = plot_parameter_correlations(expertise_data)
    fig.savefig("parameter_correlations.png")
    plt.close(fig)
    
    # Save data
    expertise_data.to_csv("expertise_data.csv", index=False)
    error_metrics.to_csv("model_errors.csv", index=False)
    
    print("\nSimulation complete. Results saved to CSV files and visualizations.")
    return expertise_data, error_metrics, fitted_models

if __name__ == "__main__":
    main()