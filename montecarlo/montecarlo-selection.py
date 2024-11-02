import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from scipy.stats import norm, lognorm, gamma, beta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_gamma_distribution_weight(sprint_points):
    """
    Calculate gamma distribution based on historical sprint points.
    
    :param sprint_points: List or numpy array of historical sprint completion points
    :return: Tuple of shape and scale parameters
    """

    # Gamma distribution
    alpha, loc, beta_scale = gamma.fit(sprint_points)
    logger.info(f"Fitting gamma distribution with alpha={alpha}, loc={loc}, beta_scale={beta_scale}")
    
    # Calculate probability weights based on the gamma distribution density
    weights = gamma.pdf(sprint_points, alpha, loc=loc, scale=beta_scale)
    weights /= np.sum(weights)
    return weights, (alpha, loc, beta_scale)

def calculate_beta_distribution_weight(sprint_points):
    """
    Calculate beta distribution based on historical sprint points.
    
    :param sprint_points: List or numpy array of historical sprint completion points
    :return: Tuple of alpha and beta parameters
    """

    # Beta distribution
    # Since Beta distribution requires data in the range [0, 1], we first scale the data
    scaled_data = (np.array(sprint_points) - min(sprint_points)) / (max(sprint_points) - min(sprint_points))
    a, b, loc, scale = beta.fit(scaled_data)
    
    weights = beta.pdf(scaled_data, a, b, loc, scale)  # Scaled data for beta
    weights /= np.sum(weights)  # Normalize weights to sum to 1
    return weights, (a, b, loc, scale)

def calculate_lognormal_distribution_weight(sprint_points):
    """
    Calculate lognormal distribution based on historical sprint points.
    
    :param sprint_points: List or numpy array of historical sprint completion points
    :return: Tuple of mean and standard deviation
    """

    # Log-normal distribution (force location to 0)
    shape, loc, scale = lognorm.fit(sprint_points, floc=0)
    logger.info(f"Fitting lognormal distribution with shape={shape}, loc={loc}, scale={scale}")
    
    # Calculate probability weights based on the lognormal distribution density
    weights = lognorm.pdf(sprint_points, shape, loc, scale)
    weights /= np.sum(weights)  # Normalize weights to sum to 1
    return weights, (shape, loc, scale)

def calculate_normal_distribution_weight(sprint_points):
    """
    Calculate normal distribution based on historical sprint points.
    
    :param sprint_points: List or numpy array of historical sprint completion points
    :return: Tuple of mean and standard deviation
    """

    # Fit a normal distribution to the historical sprint points
    mu, sigma = np.mean(sprint_points), np.std(sprint_points)
    logger.info(f"Fitting normal distribution with mean={mu} and std={sigma}")
    
    # Calculate probability weights based on the normal distribution density
    weights = norm.pdf(sprint_points, mu, sigma)
    weights /= np.sum(weights)  # Normalize weights to sum to 1

    return weights, (mu, sigma)

def monte_carlo_simulation_prob_dist(sprint_points, distribution="lognormal", num_sprints_to_simulate=5, num_simulations=1000):
    """
    Perform probability-distribution weighted Monte Carlo simulation based on historical sprint completion points.
    
    :param sprint_points: List or numpy array of historical sprint completion points
    :param num_sprints_to_simulate: Number of future sprints to simulate
    :param num_simulations: Number of Monte Carlo simulations to run
    :return: Pandas DataFrame containing simulation results and dictionary of percentiles
    """
    
       # Choose the distribution function based on input
    distribution_functions = {
        "normal": calculate_normal_distribution_weight,
        "lognormal": calculate_lognormal_distribution_weight,
        "gamma": calculate_gamma_distribution_weight,
        "beta": calculate_beta_distribution_weight
    }
    
    # Check if the provided distribution type is valid
    if distribution not in distribution_functions:
        raise ValueError(f"Invalid distribution type. Choose from: {list(distribution_functions.keys())}")
    
    
    
    # Calculate weights using the specified distribution
    weights, params = distribution_functions[distribution](sprint_points)
    logger.info(f"Using {distribution.capitalize()} distribution with parameters: {params}")
    
    logger.info(f"Starting Monte Carlo simulation with {num_simulations} iterations and probability-distribution weights.")
    
    # Store the results of each simulation
    simulations = []
    
    # Store the individual sprint results for each simulation
    individual_simulations = {f'Sprint_{i+1}': [] for i in range(num_sprints_to_simulate)}
    
    
    for i in range(num_simulations):
        # Simulate the completion points for the given number of sprints with weighted probabilities
        simulated_sprints = np.random.choice(sprint_points, size=num_sprints_to_simulate, replace=True, p=weights)
        total_points = np.sum(simulated_sprints)
        simulations.append(total_points)

        # Store each individual sprint performance in the dictionary
        for j in range(num_sprints_to_simulate):
            individual_simulations[f'Sprint_{j+1}'].append(simulated_sprints[j])
    
    # Convert results to DataFrame
    simulation_df = pd.DataFrame(simulations, columns=['Total Points'])
    individual_simulation_df = pd.DataFrame(individual_simulations)
    
    logger.info(f"Monte Carlo simulation completed. Percentiles: {simulation_df.describe()}")
    
    return simulation_df, individual_simulation_df


 #Plotting with Distribution Fits Overlay
def plot_simulation_results(total_simulation_df, individual_simulation_df, sprint_points, num_sprints_to_simulate):
    plt.figure(figsize=(10, 6))
    plt.hist(total_simulation_df['Total Points'], bins=30, color='skyblue', edgecolor='black', alpha=0.7, label="Simulated Total Points")
    
    # Overlay distribution fits
    x_vals = np.linspace(min(total_simulation_df['Total Points']), max(total_simulation_df['Total Points']), 100)
    dist_fits = {
        "Normal": calculate_normal_distribution_weight,
        "Log-Normal": calculate_lognormal_distribution_weight,
        "Gamma": calculate_gamma_distribution_weight,
        "Beta": calculate_beta_distribution_weight
    }
    for name, func in dist_fits.items():
        weights, params = func(sprint_points)
        if name == "Beta":  
            scaled_x_vals = (x_vals - min(sprint_points)) / (max(sprint_points) - min(sprint_points))
            pdf_vals = beta.pdf(scaled_x_vals, *params)
        elif name == "Normal":
            mu, sigma = params
            pdf_vals = norm.pdf(x_vals, mu, sigma)
        elif name == "Log-Normal":
            shape, loc, scale = params
            pdf_vals = lognorm.pdf(x_vals, shape, loc=loc, scale=scale)
        elif name == "Gamma":
            alpha, loc, beta_scale = params
            pdf_vals = gamma.pdf(x_vals, alpha, loc=loc, scale=beta_scale)
        
        # Scale PDF values to align with histogram
        pdf_vals *= len(total_simulation_df) * (max(x_vals) - min(x_vals)) / len(x_vals)
        plt.plot(x_vals, pdf_vals, label=f"{name} Fit")
    
    plt.title("Total Points Distribution with Distribution Fits")
    plt.xlabel("Total Points")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Individual sprint distribution plots
    for i in range(num_sprints_to_simulate):
        plt.figure(figsize=(10, 6))
        plt.hist(individual_simulation_df[f'Sprint_{i+1}'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        plt.title(f'Distribution of Points for Sprint {i+1}')
        plt.xlabel('Sprint Points')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()



# Example Usage
if __name__ == "__main__":
    # Historical sprint points
    historical_sprint_points = [20, 23, 19, 22, 24, 18, 21, 23, 20, 22]
    
    # Run Monte Carlo simulation with probability-distribution weighting
    total_simulation_df, individual_simulation_df = monte_carlo_simulation_prob_dist(
        historical_sprint_points,
        "normal",
        num_sprints_to_simulate=6,
        num_simulations=1000
    )
         
    
    # Display descriptive statistics for both total and individual performances
    logger.info("Displaying summary statistics for total group performance")
    print(total_simulation_df.describe())
    logger.info("Displaying summary statistics for individual sprint performances")
    print(individual_simulation_df.describe())
    
    # Plot results
    # plot_simulation_results(total_simulation_df, individual_simulation_df, num_sprints_to_simulate=6)
    plot_simulation_results(total_simulation_df, individual_simulation_df, historical_sprint_points, num_sprints_to_simulate=6)