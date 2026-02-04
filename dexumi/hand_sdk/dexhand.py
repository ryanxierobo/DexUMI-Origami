import pickle
from abc import ABC, abstractmethod


class DexterousHand(ABC):
    @abstractmethod
    def __init__(self):
        """Initialize the hand device."""
        pass

    @abstractmethod
    def connect(self):
        """Establish connection with the hand device."""
        pass

    @abstractmethod
    def disconnect(self):
        """Terminate connection with the hand device."""
        pass

    @abstractmethod
    def get_current_position(self):
        """Get the current position of the hand."""
        pass

    @abstractmethod
    def send_command(self, command):
        """Send a command to the hand device.

        Args:
            command: The command to send to the hand device.
        """
        pass

    @abstractmethod
    def write_hand_angle(self, angles):
        """Write angle values to the hand joints.

        Args:
            angles: The angle values to write to the hand joints.
        """
        pass


class ExoDexterousHand(DexterousHand):
    """Extended dexterous hand with additional exoskeleton-specific methods."""

    def load_model(self, model_path):
        """Load the model from the pickle file."""
        try:
            if model_path:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                print(f"Model loaded from {model_path}")
                return model
            else:
                print("No model path provided.")
                return None
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()
            return None

    @abstractmethod
    def predict_motor_value(self, joint_angle):
        """Predict the motor values from the exo encoder joint angles.

        Args:
            joint_angles: The joint angles to predict the motor values from.

        Returns:
            The predicted motor values.
        """
        pass

    def write_hand_angle_position_from_motor(self, motor_values):
        """Write the hand angle position from the motor values.

        Args:
            motor_values: The motor values to write the hand angle position from.

        Returns:
            The hand angle position from the motor values.
        """
        return self.write_hand_angle(motor_values)
