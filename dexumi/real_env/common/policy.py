import os
import pickle
import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock, Thread
from typing import Any, Optional

import numpy as np
import zmq

from .base import Request, RequestType, ZMQClientBase, ZMQServerBase


class PolicyRequestType(RequestType):
    GET_ACTION = "get_action"
    GET_OBS_CONFIG = "get_obs_config"
    GET_ATTR = "get_attr"


class PolicyServer(ZMQServerBase):
    def __init__(
        self,
        obs_config,
        pub_address: str = "ipc:///tmp/policy_pub",
        req_address: str = "ipc:///tmp/policy_req",
        max_buffer_size: int = 30,
        pub_frequency: int = 60,
        req_frequency: int = 60,
        topic: str = "policy",
        verbose: bool = True,
    ):
        """
        Initialize PolicyServer with configurable observation requirements.

        Args:
            obs_config (dict): Configuration for policy observations
            pub_address (str): Full address for PUB socket (e.g., "ipc:///tmp/policy_pub" or "tcp://127.0.0.1:5555")
            req_address (str): Full address for REQ socket (e.g., "ipc:///tmp/policy_req" or "tcp://127.0.0.1:5556")
            max_buffer_size (int): Maximum buffer size
            pub_frequency (int): Publishing frequency
            req_frequency (int): Request handling frequency
            verbose (bool): Enable verbose logging
        """
        super().__init__(
            pub_address=pub_address,
            req_address=req_address,
            topic=topic,
            max_buffer_size=max_buffer_size,
            pub_frequency=pub_frequency,
            req_frequency=req_frequency,
            verbose=verbose,
        )
        self.obs_config = obs_config
        self._validate_obs_config()

        if self.verbose:
            self._debug("PolicyServer initialized with configuration:")
            self._debug(f"Observation config: {obs_config}")
            self._debug(f"PUB address: {pub_address}")
            self._debug(f"REQ address: {req_address}")

    def start(self):
        """Start the request handling threads."""
        self._debug("Starting PolicyServer")
        self.running = True

        # For policy, we only need request handling
        self.request_thread = Thread(target=self._handle_requests)
        self.request_thread.daemon = False
        self.request_thread.start()
        self._debug("PolicyServer started successfully")

    def _validate_obs_config(self):
        """Validate the observation configuration format."""
        required_keys = {"shape", "type", "required"}

        try:
            for key, config in self.obs_config.items():
                if not isinstance(config, dict):
                    raise ValueError(f"Configuration for {key} must be a dictionary")

                missing_keys = required_keys - set(config.keys())
                if missing_keys:
                    raise ValueError(f"Missing required keys {missing_keys} for {key}")

                if not isinstance(config["shape"], tuple):
                    raise ValueError(f"Shape for {key} must be a tuple")

                if not isinstance(config["required"], bool):
                    raise ValueError(f"Required flag for {key} must be a boolean")

            self._debug("Observation configuration validated successfully")
        except Exception as e:
            print(f"Configuration validation failed: {str(e)}")
            raise

    def _check_policy_obs_shape(self, policy_obs):
        """Check if the policy observation matches the configuration."""
        try:
            if not isinstance(policy_obs, dict):
                raise ValueError("policy_obs must be a dictionary")

            # Check required observations
            for key, config in self.obs_config.items():
                if config["required"] and key not in policy_obs:
                    raise ValueError(f"Required observation '{key}' is missing")

                if key in policy_obs:
                    value = policy_obs[key]

                    # Check type
                    if config["type"] == "numpy.ndarray":
                        if not isinstance(value, np.ndarray):
                            raise ValueError(f"{key} must be a numpy array")

                        # Check shape
                        if value.shape != config["shape"]:
                            raise ValueError(
                                f"Shape mismatch for {key}. "
                                f"Expected {config['shape']}, got {value.shape}"
                            )

            self._debug("Policy observation shape check passed")
        except Exception as e:
            print(f"Shape check failed: {str(e)}")
            raise

    def _process_request(self, request):
        """Process incoming requests with enhanced error handling."""
        try:
            self._debug(f"Processing request of type: {request.type}")

            if request.type == PolicyRequestType.GET_ACTION:
                policy_obs = request.params.get("policy_obs")
                if policy_obs is None:
                    raise ValueError("Missing policy_obs in request parameters")

                action = self._predict_action(policy_obs)
                self._debug("Action prediction successful")
                return action

            elif request.type == PolicyRequestType.GET_OBS_CONFIG:
                return self.obs_config
            elif request.type == PolicyRequestType.GET_ATTR:
                attr_name = request.params.get("attr_name")
                if attr_name is None:
                    raise ValueError("Missing attr_name in request parameters")

                if not hasattr(self, attr_name):
                    raise ValueError(f"Attribute {attr_name} not found")

                return getattr(self, attr_name)

            print(f"Unknown request type: {request.type}")
            return {"error": f"Unknown request type: {request.type}"}

        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return {"error": str(e)}

    def _predict_action(self, policy_obs):
        """Predict action from policy observation with enhanced error handling."""
        try:
            self._debug("Starting action prediction")
            self._check_policy_obs_shape(policy_obs)

            processed_obs = self._preprocess_policy_obs(policy_obs)
            self._debug("Observation preprocessing complete")

            action = self._inference_action(processed_obs)
            self._debug("Action inference complete")

            return action

        except Exception as e:
            print(f"Error in action prediction: {str(e)}")
            raise

    def _preprocess_policy_obs(self, policy_obs):
        """Override this method to preprocess policy observation"""
        raise NotImplementedError("Subclass must implement _preprocess_policy_obs()")

    def _inference_action(self, policy_obs):
        """Override this method to implement action inference"""
        raise NotImplementedError("Subclass must implement _inference_action()")

    @classmethod
    def create_with_example_input(cls, example_input: dict, **kwargs):
        """Create a PolicyServer instance with configuration derived from example input."""
        obs_config = {}

        for key, value in example_input.items():
            if isinstance(value, np.ndarray):
                obs_config[key] = {
                    "shape": value.shape,
                    "type": "numpy.ndarray",
                    "required": True,
                }
            else:
                raise ValueError(f"Unsupported input type for {key}: {type(value)}")

        return cls(obs_config=obs_config, **kwargs)


class PolicyClient(ZMQClientBase):
    def __init__(
        self,
        pub_address: str = "ipc:///tmp/policy_pub",
        req_address: str = "ipc:///tmp/policy_req",
        req_frequency: int = 1000,
        topic: str = "policy",
        verbose: bool = True,
    ):
        """Initialize PolicyClient with flexible transport support."""
        super().__init__(
            pub_address=pub_address,
            req_address=req_address,
            topic=topic,
            req_frequency=req_frequency,
            verbose=verbose,
        )
        self._debug("PolicyClient initialized")
        try:
            self.model_cfg = self.get_attr("model_cfg")
        except Exception as e:
            print(f"Error getting model config: {str(e)}")

    def get_action(self, policy_obs, timeout=1.0):
        """Get action from policy server with timeout support."""
        try:
            self._debug("Sending get_action request")
            request = Request(
                type=PolicyRequestType.GET_ACTION, params={"policy_obs": policy_obs}
            )

            response = self.send_request(request, timeout=timeout)
            self._debug("Received action response")

            return response

        except TimeoutError:
            print("Timeout while waiting for action response")
            raise
        except Exception as e:
            print(f"Error in get_action: {str(e)}")
            raise

    def get_obs_config(self, timeout=1.0):
        """Get observation configuration from server."""
        try:
            self._debug("Requesting observation configuration")
            request = Request(type=PolicyRequestType.GET_OBS_CONFIG)
            return self.send_request(request, timeout=timeout)
        except Exception as e:
            print(f"Error getting observation config: {str(e)}")
            raise

    def get_attr(self, attr_name, timeout=1.0):
        """Get attribute from server."""
        try:
            self._debug(f"Requesting attribute: {attr_name}")
            request = Request(
                type=PolicyRequestType.GET_ATTR, params={"attr_name": attr_name}
            )
            return self.send_request(request, timeout=timeout)
        except Exception as e:
            print(f"Error getting attribute: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Example 1: Using default configuration
    server1 = PolicyServer()

    # Example 2: Using custom configuration
    custom_config = {
        "visual": {
            "shape": (1, 480, 640, 3),
            "type": "numpy.ndarray",
            "required": True,
        },
        "robot_state": {"shape": (1, 7), "type": "numpy.ndarray", "required": True},
    }
    server2 = PolicyServer(obs_config=custom_config)

    # Example 3: Creating from example input
    example_input = {
        "visual": np.zeros((1, 224, 224, 3)),
        "robot_state": np.zeros((1, 7)),
    }
    server3 = PolicyServer.create_with_example_input(example_input)

    # Example client usage
    client = PolicyClient()

    # Example policy observation
    policy_obs = {
        "visual_obs": np.random.uint8(np.zeros((1, 224, 224, 3))),
        "robot_state": np.zeros((1, 7)),
    }

    # Get action from server
    action = client.get_action(policy_obs)
