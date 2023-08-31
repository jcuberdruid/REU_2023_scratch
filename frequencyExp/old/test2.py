import numpy as np
import matplotlib.pyplot as plt

def perform_2d_fourier_transform(image_641x64):
    # Check dimensions
    if not all(len(row) == 64 for row in image_641x64) or len(image_641x64) != 641:
        raise ValueError("Input should be a 641x64 array.")
    
    # Convert list of lists to numpy array for better performance
    image_array = np.array(image_641x64)
    
    # Perform 2D Fourier Transform
    fourier_transform = np.fft.fft2(image_array)
    
    # Shift the zero frequency component to the center
    fourier_transform_shifted = np.fft.fftshift(fourier_transform)
    
    # Take the magnitude of the Fourier coefficients for display
    magnitude_spectrum = np.abs(fourier_transform_shifted)

    return magnitude_spectrum

# Generate a sample image: 641x64 with a simple pattern
sample_image = np.zeros((641, 64))
sample_image[300:341, 28:36] = 1

# Perform 2D Fourier Transform
result = perform_2d_fourier_transform(sample_image.tolist())

# Display the original and Fourier-transformed images
plt.figure()

plt.subplot(121)
plt.imshow(sample_image, cmap='gray')
plt.title('Original Image')
plt.colorbar()

plt.subplot(122)
plt.imshow(np.log1p(result), cmap='gray')  # log1p used for better visibility
plt.title('Fourier Transform')
plt.colorbar()

plt.show()

