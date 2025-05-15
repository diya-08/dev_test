# file: assessment/forecast.py
import pandas as pd
import argparse
from prophet import Prophet
import matplotlib.pyplot as plt


def load_data(filepath: str) -> pd.DataFrame:
    """Load and return weather dataset from a CSV file."""
    # Ensure that the CSV file exists
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError("The CSV file does not exist.")

    # Raise KeyErrors for each column if the CSV file does not contain them
    if 'ds' not in df.columns:
        raise KeyError("The CSV file does not contain the 'ds' column.")
    if 'temperature_celsius' not in df.columns:
        raise KeyError("The CSV file does not contain the 'temperature_celsius' column.")
    if 'humidity' not in df.columns:
        raise KeyError("The CSV file does not contain the 'humidity' column.")

    # Ensure that the 'ds' column is a valid datetime format, if not it will return 'NaT'
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

    return df


def train_model(df: pd.DataFrame, use_regressor: bool = False, tuned_model: bool = False) -> Prophet:
    """Train a basic Prophet model. Extend with tuning if needed."""
    model = Prophet()

    # TODO: Add external regressor  humidity
    if use_regressor:
        model.add_regressor('humidity')
    # TODO: Replace with tuned Prophet model using Optuna
    if tuned_model:
        possible_parameters = [
            # The additive seasonality mode: the overall
            # weather pattern is not going to change much over time (stable trend).
            # Hence, we don't expect any big shifts in the usual weather patterns.
            {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
            # The multiplicative seasonality mode: the overall weather pattern to
            # change more easily over time (flexible trend).
            # Hence, we might expect bigger shifts in weather patterns maybe because of climate change, for example.
            {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 5.0, 'seasonality_mode': 'multiplicative'}
        ]

        # Setting the Mean Absolute Percentage Error to infinity because the error will be smaller than infinity
        best_mape = float('inf')
        # Created an empty dictionary that will store the set that had the best results for best_mape (i.e: the lowest error)
        best_parameters = {}

        for parameter_set in possible_parameters:
            # Created a temporary Prophet model with three parameters: 'changepoint_prior_scale', 'seasonality_prior_scale', 'seasonality_mode'
            temporary_model = Prophet(
                changepoint_prior_scale = parameter_set['changepoint_prior_scale'],
                seasonality_prior_scale = parameter_set['seasonality_prior_scale'],
                seasonality_mode = parameter_set['seasonality_mode']
            )
            # Added the humidity regressor for the temporary model if it is selected
            if use_regressor:
                temporary_model.add_regressor('humidity')

            # Trains the temporary model: The independent/input variable is 'ds' column and the dependent/target variable is 'y' 
            # which is 'temperature_celsius' column and the 'humidity' column will be added if 'use_regressor' is 'True'
            temporary_model.fit(df[['ds', 'y'] + (['humidity'] if use_regressor else [])])
            # Performs cross validation on the temporary model
            df_cross_validation = cross_validation(temporary_model, initial='210 days', period='30 days', horizon='30 days', parallel='processes')
            # Gives performance metrics including MAPE
            df_performance_metrics = performance_metrics(df_cross_validation)
            # Calculates the mean of the MAPE values
            current_mape = df_performance_metrics['mape'].mean()

            print(f"The attempted parameters: {parameter_set}, MAPE: {current_mape}")

            # Finds the lowest Mean Average Percentage Error and tracks the best parameters
            if current_mape < best_mape:
                best_mape = current_mape
                best_parameters = parameter_set

        # Prints out and creates the final model with the best parameters and the Mean Average Percentage Error
        print(f"The best parameters: {best_parameters} MAPE: {best_mape}")
        model = Prophet(
                changepoint_prior_scale = best_parameters['changepoint_prior_scale'],
                seasonality_prior_scale = best_parameters['seasonality_prior_scale'],
                seasonality_mode = best_parameters['seasonality_mode']
        )

        # Added the humidity regressor for the final model if it is selected
        if use_regressor:
            model.add_regressor('humidity')

    # Trains the final model
    model.fit(df[['ds', 'y'] + (['humidity'] if use_regressor else [])])
    return model


def make_future_dataframe(model: Prophet, df: pd.DataFrame, periods: int, use_regressor: bool = False) -> pd.DataFrame:
    """Create future dataframe for forecasting."""
    future = model.make_future_dataframe(periods=periods)

    # TODO: Add regressor values to future dataframe if used

    # Creates a column in the 'future' dataframe called 'humidity' 
    # and fills it with the average humidity value that 
    # was calculated from the historical data
    if use_regressor:
    future['humidity'] = df['humidity'].mean()

    return future


def generate_forecast(model: Prophet, future: pd.DataFrame) -> pd.DataFrame:
    """Generate forecast using the model."""
    forecast = model.predict(future)

    # Raise ValueErrors for the missing forecast columns
    if 'ds' not in forecast.columns:
        raise ValueError("The 'ds' forecast column is missing.")
    if 'yhat' not in forecast.columns:
        raise ValueError("The 'yhat' forecast column is missing.")
    if 'yhat_lower' not in forecast.columns:
        raise ValueError("The 'yhat_lower' forecast column is missing.")
    if 'yhat_upper' not in forecast.columns:
        raise ValueError("The 'yhat_upper' forecast column is missing.")

    return forecast


def plot_forecast(df: pd.DataFrame, forecast: pd.DataFrame, output_path: str = None):
    """Visualize the forecast vs. actuals.

    TODO: Implement this function to plot results using matplotlib or plotly
    """

    marker_colour_actual = 'aquamarine'
    line_colour_forecast = 'red'
    fill_colour = 'lightcoral'
    
    # Create a new figure with a defined size
    plt.figure(figsize=(10, 6))

    # Plot observed historical data values as 'o'
    plt.plot(df['ds'], df['y'], marker='o', color=marker_colour_actual, label='Actual Temps')

    # Plot the forecasted values as a line '-'
    plt.plot(forecast['ds'], forecast['yhat'], linestyle='-', color=line_colour_forecast, label='Predicted Temps')

    # Shade the area between lower and upper prediction intervals and added a label 'uncertainty' for the fill area
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color=fill_colour, alpha=0.5, label='Uncertainty')

    # Labelled the x and y axes, added a title and displayed the legend to the plot
    plt.xlabel('Date', fontsize=14, fontweight = 'bold')
    plt.ylabel('Temperature (Degrees Celsius)', fontsize=14, fontweight = 'bold')
    plt.title('Weather Forecast', fontsize=16, fontweight = 'bold')
    plt.legend()

    # Adjusts the labels and title to fit neatly
    plt.tight_layout()

    # If an output path is provided, it will save the figure otherwise it will display the figure
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Weather Forecast using Prophet")

    # TODO: Add CLI arguments:
    # --input: path to input CSV file
    # --periods: forecast horizon
    # --output: path to output CSV
    # --use_regressor: flag to use humidity
    # parser.add_argument('--input', ...)
    args = parser.parse_args()

    df = load_data(args.input)

    # Rename the target variable for Prophet
    df.rename(columns={'temperature_celsius': 'y'}, inplace=True)

    model = train_model(df)
    future = make_future_dataframe(model, df, args.periods)
    forecast = generate_forecast(model, future)

    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # TODO: Call plot_forecast and save CSV if output provided
    # plot_forecast(df, forecast)
    # forecast.to_csv(args.output)


if __name__ == "__main__":
    main()
