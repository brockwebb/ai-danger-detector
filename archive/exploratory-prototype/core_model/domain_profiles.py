"""
Domain-specific parameter profiles for the expertise-harm model.
"""

domain_profiles = {
    "medical": {
        "name": "Medical Diagnosis",
        "description": "AI systems used for disease diagnosis, treatment recommendations, or health monitoring",
        "E_min": 2.0,  # Higher baseline expertise needed
        "beta": [0.4, 0.2, 0.2, 0.15],  # Higher sensitivity to harm
        "alpha": [1.8, 1.1, 0.9, 1.2]   # Steeper curve for harm
    },
    
    "legal": {
        "name": "Legal Analysis",
        "description": "AI systems for legal research, contract analysis, or case outcome prediction",
        "E_min": 2.0,  # Higher baseline expertise needed
        "beta": [0.2, 0.4, 0.2, 0.15],  # Higher sensitivity to complexity
        "alpha": [1.2, 1.7, 0.8, 1.1]   # Steeper curve for complexity
    },
    
    "creative": {
        "name": "Creative Writing",
        "description": "AI systems for content creation, storytelling, or creative assistance",
        "E_min": 1.0,  # Lower baseline expertise needed
        "beta": [0.1, 0.2, 0.1, 0.05],  # Lower sensitivity overall
        "alpha": [1.1, 1.0, 0.7, 0.9]   # Gentler curves
    },
    
    "statistical": {
        "name": "Data Analysis",
        "description": "AI systems for statistical analysis, data visualization, or modeling",
        "E_min": 1.5,  # Moderate baseline expertise needed
        "beta": [0.2, 0.3, 0.3, 0.1],  # Higher sensitivity to error rates
        "alpha": [1.3, 1.4, 1.2, 1.0]  # Steeper curve for complexity and error
    },
    
    "financial": {
        "name": "Financial Advisory",
        "description": "AI systems for investment recommendations, financial planning, or risk assessment",
        "E_min": 2.0,  # Higher baseline expertise needed
        "beta": [0.3, 0.3, 0.2, 0.2],  # Balanced sensitivity to harm and complexity
        "alpha": [1.5, 1.5, 1.0, 1.2]  # Steep curves for both harm and complexity
    },
    
    "educational": {
        "name": "Educational Content",
        "description": "AI systems for education, tutoring, or knowledge dissemination",
        "E_min": 1.5,  # Moderate baseline expertise needed
        "beta": [0.2, 0.3, 0.15, 0.1],  # Higher sensitivity to complexity
        "alpha": [1.2, 1.3, 0.8, 1.0]   # Moderate curves
    }
}