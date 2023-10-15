import numpy as np
import matplotlib.pyplot as plt
import plotly
import scipy.stats as stats

class InvestmentSimulation:
    def __init__(self, num_people, interest_rate, scaling_factor, iteration_time):
        self.num_people = num_people
        self.interest_rate = interest_rate
        self.scaling_factor = scaling_factor
        self.iteration_time = iteration_time
        self.wealth_distribution = None

    def assign_investment_types(self):
        # Define the investment strategies
        strategies = [
            {"name": "Conservative", "ratio": 0.1},
            {"name": "Moderate 1", "ratio": 0.2},
            {"name": "Moderate 2", "ratio": 0.3},
            {"name": "Moderate 3", "ratio": 0.4},
            {"name": "Moderate 4", "ratio": 0.5},
            {"name": "Aggressive 1", "ratio": 0.6},
            {"name": "Aggressive 2", "ratio": 0.7},
            {"name": "Aggressive 3", "ratio": 0.8},
            {"name": "Aggressive 4", "ratio": 0.9},
            {"name": "Aggressive 5", "ratio": 1.0},
        ]

        # Calculate the Pareto weights based on strategy ratios
        ratios = np.array([strategy["ratio"] for strategy in strategies])
        weights = 1 / ratios
        pareto_weights = weights / np.sum(weights)

        # Use the Pareto weights to assign investment strategies to individuals
        investment_strategy = np.random.choice(len(strategies), size=self.num_people, p=pareto_weights)
        
        self.investment_ratio = np.array([strategies[strategy]["ratio"] for strategy in investment_strategy])

        return self.investment_ratio

    def simulate_society(self):
        # Initialize the society
        m = 100.0  # Starting money for each person
        wealth = np.full(self.num_people, m)  # Initial wealth for all people

        # Generate investment strategy type for each person using random choice
        investment_ratio = self.assign_investment_types()

        # Lists to store wealth distribution after each round
        wealth_distribution = [wealth.copy()]

        # Run simulation for 'iteration_time' rounds
        for _ in range(self.iteration_time):
            # Calculate total consumption amount for the round
            total_consumption = np.sum((1 - investment_ratio) * wealth)

            # Calculate investment and consumption amount for each person
            investment_amount = investment_ratio * wealth
            consumption_amount = np.full(self.num_people, total_consumption / self.num_people)

            # Simulate investment and update wealth
            investment_returns = np.random.normal(self.interest_rate, self.scaling_factor * self.interest_rate, self.num_people)
            wealth += investment_amount * investment_returns

            # Calculate consumption and update wealth
            #wealth -= investment_amount
            wealth += consumption_amount

            # Ensure wealth is non-negative
            wealth = np.maximum(wealth, 0)

            # Store the wealth distribution after each round
            wealth_distribution.append(wealth.copy())

        self.wealth_distribution = wealth_distribution

    def plot_wealth_distribution(self):

        fig = plotly.graph_objects.Figure()

        # Add traces, one for each slider step
        for wealth in self.wealth_distribution:
            fig.add_trace(
                plotly.graph_objects.Histogram(x=wealth, nbinsx=100, histnorm='probability density', visible=False))
            
        # Make 10th trace visible
        fig.data[10].visible = True

        # Create and add slider
        steps = []
        for i in range(len(fig.data)):

            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                    {"title": "Wealth Distribution for Round " + str(i)}],  # layout attribute
            )
            step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        sliders = [dict(
            active=10,
            currentvalue={"prefix": "Round: "},
            pad={"t": 50},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders,
            width=800,
            height=600,
        )

        fig.show()

    # plot the normal distribution interest return

