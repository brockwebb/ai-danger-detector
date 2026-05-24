"""
Core model definition for the multi-regime expertise-harm model.
"""

def expertise_required(h, c, e, E_min=1.0, beta=[0.2, 0.3, 0.2, 0.1], alpha=[1.5, 1.2, 0.8, 1.1]):
    """
    Calculate required expertise level based on input parameters
    
    Parameters:
    -----------
    h : float
        Harm potential (scale: 1-10)
        Represents the potential negative consequences if AI output is incorrect
    
    c : float
        Complexity (scale: 1-10)
        Represents the domain knowledge depth and interconnectedness
    
    e : float
        Error rate (scale: 0-1)
        Represents the AI model's hallucination or mistake frequency (e.g., 0.15 = 15%)
    
    E_min : float
        Minimum expertise baseline (scale: 1-10)
        The baseline expertise required even for low-risk scenarios
    
    beta : list of 4 floats
        Scaling coefficients for [harm, complexity, error, interaction]
        Controls how sensitive expertise requirements are to each parameter
    
    alpha : list of 4 floats
        Exponents controlling curve shapes for [harm, complexity, error, interaction]
        Higher values create steeper curves (faster expertise growth)
    
    Returns:
    --------
    float
        Required expertise level (scale: 1-10)
    """
    # Individual contributions
    harm_component = beta[0] * (h ** alpha[0])
    complexity_component = beta[1] * (c ** alpha[1])
    error_component = beta[2] * (e ** alpha[2])
    
    # Interaction effect (how parameters multiply each other's effects)
    interaction = beta[3] * ((h * c * e) ** alpha[3])
    
    # Total expertise required (capped at maximum of 10)
    expertise = E_min + harm_component + complexity_component + error_component + interaction
    return min(expertise, 10.0)