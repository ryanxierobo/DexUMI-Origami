import time

import cv2
import numpy as np
import qrcode
from pyzbar.pyzbar import decode


# Function to generate a QR code with encoded monotonic time
def generate_qr(data):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=8,
        border=8,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white")

    # Convert the PIL image to OpenCV format (numpy array)
    open_cv_image = np.array(img.convert("RGB"))
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    return open_cv_image


def display_qr_cv2(img):
    cv2.imshow("QR Code", img)
    cv2.waitKey(1)  # Display for 1 millisecond to allow updating


def dynamic_qr_code():
    try:
        while True:
            # Get current monotonic time as a float
            monotonic_time = time.monotonic()

            # Encode monotonic time into QR code as a string
            dynamic_data = f"Monotonic Time: {monotonic_time}"

            # Generate the QR code with the monotonic time as data
            qr_img = generate_qr(dynamic_data)

            # Display the QR code using OpenCV
            display_qr_cv2(qr_img)

            # Print the monotonic time and the encoded data for demonstration
            print(f"QR Code generated with monotonic time: {monotonic_time}")

            # Wait for 0.02 seconds before generating the next QR code
            time.sleep(1 / 120)

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        # Close the OpenCV window when done
        cv2.destroyAllWindows()


# Function to read and decode QR codes from an image
def read_time_from_qr_code(image):
    # Use pyzbar to decode the QR code(s) in the image
    decoded_objects = decode(image)

    # Process each QR code found in the image
    for obj in decoded_objects:
        # Extract the data encoded in the QR code
        qr_data = obj.data.decode("utf-8")

        # Print the full data for reference
        print(f"QR Code Data: {qr_data}")

        # Assuming that the data contains 'Monotonic Time: <value>', extract the time
        if "Monotonic Time:" in qr_data:
            monotonic_time = float(qr_data.split("Monotonic Time:")[1].strip())
            # print(f"Extracted Monotonic Time: {monotonic_time}")
            return monotonic_time
        else:
            print("No monotonic time found in the QR code data.")

    return None
