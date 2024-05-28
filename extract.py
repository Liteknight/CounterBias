import SimpleITK as sitk
import os

dir = "./exp140/"
split = "val/"

output_dir = "./far_bias/" + split

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file in os.listdir(dir + split):
    if file.endswith(".nii.gz"):
        try:
            # Read the image
            image = sitk.ReadImage(os.path.join(dir, split, file))

            # Extract a slice along the z-axis at index 75
            image_slice = image[:, :, 75]

            # Construct the output file path with ".tif" extension
            output_file_path = os.path.join(output_dir, file.replace(".nii.gz", ".tiff"))

            # Write the extracted slice to the output directory in TIFF format
            sitk.WriteImage(image_slice, output_file_path)

        except Exception as e:
            print(f"Error processing file {file}: {e}")
