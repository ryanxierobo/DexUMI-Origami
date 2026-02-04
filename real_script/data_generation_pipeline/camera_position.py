import cv2
import cv2.aruco as aruco
import numpy as np

from dexumi.common.utility.matrix import invert_transformation


def get_camera_position_and_orientation(frame_data, marker_size, viz=False):
    try:
        # Camera calibration parameters
        camera_matrix = frame_data.intrinsics
        dist_coeffs = np.zeros((4, 1))  # Assuming no distortion

        # Get RGB and depth images
        rgb_image = frame_data.rgb
        depth_image = frame_data.depth

        if rgb_image is None or depth_image is None:
            print("Error: RGB or depth image is None")
            return

        # Convert RGB to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # Create ArUco dictionary and detect markers
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)
        if viz:
            for i, corner in enumerate(corners):
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
                for j, point in enumerate(corner[0]):
                    point_int = tuple(point.astype(int))
                    cv2.circle(rgb_image, point_int, 5, colors[j], -1)

                    # Draw the corner index
                    cv2.putText(
                        rgb_image,
                        str(j),
                        (point_int[0] + 10, point_int[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        colors[j],
                        2,
                    )
            cv2.imshow("ArUco Markers", rgb_image)

        if ids is not None and len(ids) > 0:
            for i in range(len(ids)):
                # Get the marker corners
                marker_corners = corners[i][0]

                # 3D coordinates for the marker corners
                marker_3d_points = []
                valid_corners = []

                for corner in marker_corners:
                    corner_x = int(corner[0])
                    corner_y = int(corner[1])

                    # Ensure depth is valid (non-zero and non-NaN)
                    corner_depth = depth_image[corner_y, corner_x]
                    if corner_depth > 0 and not np.isnan(corner_depth):
                        # Project corner from image space to 3D world space
                        x_3d = (
                            (corner_x - camera_matrix[0, 2])
                            * corner_depth
                            / camera_matrix[0, 0]
                        )
                        y_3d = (
                            (corner_y - camera_matrix[1, 2])
                            * corner_depth
                            / camera_matrix[1, 1]
                        )
                        z_3d = corner_depth  # Depth gives us the Z coordinate directly

                        marker_3d_points.append([x_3d, y_3d, z_3d])
                        valid_corners.append(corner)

                marker_3d_points = np.array(marker_3d_points, dtype=np.float32)
                valid_corners = np.array(valid_corners, dtype=np.float32)

                # Ensure we have at least 4 valid 3D-2D correspondences
                if len(marker_3d_points) < 4:
                    print(
                        "Error: Not enough valid depth points for marker pose estimation"
                    )
                    continue

                # Define the reference 3D positions of the marker's corners in the marker's own coordinate system
                marker_size_half = marker_size / 2.0
                reference_3d_corners = np.array(
                    [
                        [-marker_size_half, marker_size_half, 0],
                        [marker_size_half, marker_size_half, 0],
                        [marker_size_half, -marker_size_half, 0],
                        [-marker_size_half, -marker_size_half, 0],
                    ],
                    dtype=np.float32,
                )

                # Use solvePnP to estimate the rotation and translation from the 3D points
                success, rvec, tvec = cv2.solvePnP(
                    reference_3d_corners, valid_corners, camera_matrix, dist_coeffs
                )

                if success:
                    # Convert the rotation vector to a rotation matrix
                    rotation_matrix, _ = cv2.Rodrigues(rvec)

                    # Invert the rotation and translation to get the camera's pose relative to the marker
                    T_CA = np.eye(4)
                    T_CA[:3, :3] = rotation_matrix
                    T_CA[:3, -1] = tvec[:, 0]
                    T_CA[3, 3] = 1
                    T_AC = invert_transformation(T_CA)
                    # Draw axis for visualization
                    cv2.drawFrameAxes(
                        rgb_image, camera_matrix, dist_coeffs, rvec, tvec, 0.05
                    )
                    return T_AC, rgb_image
                else:
                    print("Error: solvePnP failed")
                    return None, None

        else:
            print("No ArUco markers detected")
            return None, None

    except Exception as e:
        print(f"Error in get_camera_position_and_orientation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    from dexumi.camera.iphone_camera import IphoneCamera

    app = IphoneCamera()
    app.start_streaming()

    try:
        while True:
            frame_data = app.get_camera_frame()
            if frame_data.rgb is not None:
                cv2.imshow("RGB", frame_data.rgb)
                get_camera_position_and_orientation(
                    frame_data, (155 + 32.5) / 1000
                )  # Assuming marker size is 10cm

            if frame_data.depth is not None:
                print(frame_data.depth.shape)
                print(frame_data.depth.max())

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        app.stop_streaming()
        cv2.destroyAllWindows()
