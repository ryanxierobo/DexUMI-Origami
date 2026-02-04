import json
import os
import pickle
from functools import partial
from multiprocessing import Pool

import click
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def create_trajectory_visualization(real_data, sim_data, finger_name):
    """
    Create interactive 3D visualization of real and simulated trajectories
    with different marker styles and separated legend/colorbars
    """
    # Extract data
    real_trajectory = real_data[finger_name]["T_flange_cubes"][:, :3, 3]
    sim_trajectory = sim_data[finger_name]["trajectory"][:, :3, 3]
    motor_values = real_data[finger_name]["motor_value"] / 1000
    joint_values = np.array(sim_data[finger_name]["joint_values"]).flatten()

    # Create figure
    fig = go.Figure()

    # Add real trajectory
    fig.add_trace(
        go.Scatter3d(
            x=real_trajectory[:, 0],
            y=real_trajectory[:, 1],
            z=real_trajectory[:, 2],
            mode="markers",
            name="Real Trajectory",
            marker=dict(
                size=4,
                symbol="circle",
                color=motor_values,
                colorscale="Viridis",
                opacity=0.8,
                colorbar=dict(
                    title="Motor Value",
                    x=0.85,
                    y=0.5,
                    len=0.75,
                ),
            ),
            hovertemplate=(
                "X: %{x:.3f}<br>"
                + "Y: %{y:.3f}<br>"
                + "Z: %{z:.3f}<br>"
                + "Motor Value: %{marker.color:.3f}<br>"
                + "<extra></extra>"
            ),
        )
    )

    # Add simulated trajectory
    fig.add_trace(
        go.Scatter3d(
            x=sim_trajectory[:, 0],
            y=sim_trajectory[:, 1],
            z=sim_trajectory[:, 2],
            mode="markers",
            name="Sim Trajectory",
            marker=dict(
                size=4,
                symbol="diamond",
                color=joint_values,
                colorscale="Plasma",
                opacity=0.8,
                colorbar=dict(
                    title="Joint Value",
                    x=1.0,
                    y=0.5,
                    len=0.75,
                ),
            ),
            hovertemplate=(
                "X: %{x:.3f}<br>"
                + "Y: %{y:.3f}<br>"
                + "Z: %{z:.3f}<br>"
                + "Joint Value: %{marker.color:.3f}<br>"
                + "<extra></extra>"
            ),
        )
    )

    # Update layout
    fig.update_layout(
        title="Real vs Simulated Trajectory",
        scene=dict(
            aspectmode="data",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            zaxis_title="Z Position",
        ),
        width=1200,
        height=800,
        showlegend=True,
        legend=dict(
            x=0.7,
            y=0.9,
            xanchor="right",
            yanchor="top",
        ),
        margin=dict(r=100),
    )

    return fig


def match_trajectories_and_regress(
    real_data, sim_data, finger_name, polynomial_degree=3
):
    """
    Match points between real and simulated trajectories and perform regression
    between joint angles and motor values.
    """
    # Extract real trajectory points and motor values
    real_trajectory = real_data[finger_name]["T_flange_cubes"][:, :3, 3]
    motor_values = real_data[finger_name]["motor_value"] / 1000

    # Extract simulated trajectory points and joint values
    sim_trajectory = sim_data[finger_name]["trajectory"][:, :3, 3]
    joint_values = np.array(sim_data[finger_name]["joint_values"]).flatten()

    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(sim_trajectory)
    distances, indices = nbrs.kneighbors(real_trajectory)

    # Extract matched joint values and motor values
    matched_joints = joint_values[indices.flatten()]

    # Create and fit regression model
    model = make_pipeline(
        PolynomialFeatures(degree=polynomial_degree, include_bias=False),
        LinearRegression(),
    )

    X = matched_joints.reshape(-1, 1)
    model.fit(X, motor_values)

    # Make predictions
    predictions = model.predict(X)

    # Calculate metrics
    r2 = r2_score(motor_values, predictions)
    mae = mean_absolute_error(motor_values, predictions)

    # Create visualization
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Regression Analysis", "Matching Quality"),
        specs=[[{"type": "scatter"}, {"type": "scatter3d"}]],
        horizontal_spacing=0.1,
    )

    # Add regression plot
    fig.add_trace(
        go.Scatter(
            x=matched_joints,
            y=motor_values,
            mode="markers",
            name="Actual Values",
            marker=dict(size=8, color="rgb(70, 130, 180)", opacity=0.6),
            hovertemplate=(
                "Joint Value: %{x:.3f}<br>"
                + "Motor Value: %{y:.3f}<br>"
                + "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # Add regression line
    joint_range = np.linspace(matched_joints.min(), matched_joints.max(), 100)
    predicted_range = model.predict(joint_range.reshape(-1, 1))

    fig.add_trace(
        go.Scatter(
            x=joint_range,
            y=predicted_range,
            mode="lines",
            name="Regression Line",
            line=dict(color="red", width=2),
        ),
        row=1,
        col=1,
    )

    # Add 3D matching visualization with enhanced hover information
    fig.add_trace(
        go.Scatter3d(
            x=real_trajectory[:, 0],
            y=real_trajectory[:, 1],
            z=real_trajectory[:, 2],
            mode="markers",
            name="Real Trajectory",
            marker=dict(size=4, color="blue", opacity=0.6),
            customdata=np.column_stack(
                [motor_values, matched_joints, distances.flatten()]
            ),
            hovertemplate=(
                "X: %{x:.3f}<br>"
                + "Y: %{y:.3f}<br>"
                + "Z: %{z:.3f}<br>"
                + "Motor Value: %{customdata[0]:.3f}<br>"
                + "Matched Joint Value: %{customdata[1]:.3f}<br>"
                + "Matching Distance: %{customdata[2]:.3f}m<br>"
                + "<extra>Real Point</extra>"
            ),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter3d(
            x=sim_trajectory[indices.flatten()][:, 0],
            y=sim_trajectory[indices.flatten()][:, 1],
            z=sim_trajectory[indices.flatten()][:, 2],
            mode="markers",
            name="Matched Sim Points",
            marker=dict(size=4, color="red", opacity=0.6),
            customdata=np.column_stack(
                [motor_values, matched_joints, distances.flatten()]
            ),
            hovertemplate=(
                "X: %{x:.3f}<br>"
                + "Y: %{y:.3f}<br>"
                + "Z: %{z:.3f}<br>"
                + "Motor Value: %{customdata[0]:.3f}<br>"
                + "Joint Value: %{customdata[1]:.3f}<br>"
                + "Matching Distance: %{customdata[2]:.3f}m<br>"
                + "<extra>Matched Sim Point</extra>"
            ),
        ),
        row=1,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title="Joint to Motor Value Regression with Trajectory Matching",
        width=1500,
        height=700,
        template="plotly_white",
        showlegend=True,
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"R² = {r2:.3f}<br>MAE = {mae:.3f}<br>Mean matching distance = {distances.mean():.3f}m",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )

    # Update axis labels
    fig.update_xaxes(title_text="Joint Values", row=1, col=1)
    fig.update_yaxes(title_text="Motor Values", row=1, col=1)

    # Update 3D scene
    fig.update_scenes(
        aspectmode="data",
        xaxis_title="X Position",
        yaxis_title="Y Position",
        zaxis_title="Z Position",
        row=1,
        col=2,
    )

    metrics = {"r2": r2, "mae": mae, "mean_matching_distance": distances.mean()}

    return model, metrics, fig


@click.command()
@click.option(
    "-r",
    "--real-traj",
    default="real_fingertip_trajectory.pkl",
    help="Path to real trajectory pickle file",
)
@click.option(
    "-s",
    "--sim-traj",
    default="sim_fingertip_trajectory.pkl",
    help="Path to simulation trajectory pickle file",
)
@click.option(
    "-o",
    "--output-path",
    default="encoder_to_motor_regression",
    help="Path to save output files",
)
@click.option(
    "-p",
    "--polynomial-degree",
    type=int,
    default=3,
    show_default=True,
    help="Degree of polynomial regression",
)
@click.option(
    "-n",
    "--finger-name",
    default="index",
    help="Name of the finger to analyze",
)
def main(real_traj, sim_traj, output_path, polynomial_degree, finger_name):
    """
    Main function to perform trajectory matching and regression analysis.
    """
    # Load data
    print("Loading trajectory data...")
    with open(real_traj, "rb") as f:
        real_fingertips_trajectory = pickle.load(f)

    with open(sim_traj, "rb") as f:
        sim_fingertips_trajectory = pickle.load(f)

    # Create and show trajectory visualization
    print("Creating trajectory visualization...")
    traj_fig = create_trajectory_visualization(
        real_fingertips_trajectory, sim_fingertips_trajectory, finger_name
    )
    traj_fig.show()

    # Perform matching and regression
    print("Performing trajectory matching and regression...")
    model, metrics, reg_fig = match_trajectories_and_regress(
        real_fingertips_trajectory,
        sim_fingertips_trajectory,
        finger_name,
        polynomial_degree,
    )

    # Show regression plot
    reg_fig.show()

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Save the model
    model_filename = os.path.join(output_path, f"joint_to_motor_{finger_name}.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

    # Save metrics
    metrics_filename = os.path.join(output_path, f"metrics_{finger_name}.json")
    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nResults for {finger_name} finger:")
    print(f"R² score: {metrics['r2']:.3f}")
    print(f"Mean Absolute Error: {metrics['mae']:.3f}")
    print(f"Mean matching distance: {metrics['mean_matching_distance']:.3f}m")
    print(f"\nModel saved to: {model_filename}")
    print(f"Metrics saved to: {metrics_filename}")


if __name__ == "__main__":
    main()
