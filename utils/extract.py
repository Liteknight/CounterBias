"""
Extracts 2D image slices from 3D images
"""

import SimpleITK as sitk
import os

dir = "./../exp201_intensity_only/"

for split in ["train", "val", "test"]:
    output_dir = "./../int_bias/" + split

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Extracting slices from " + split + "...")

    for file in os.listdir(dir + split):
        if file.endswith(".nii.gz"):
            try:
                # Read the image
                image = sitk.ReadImage(os.path.join(dir, split, file))

                # Extract a slice along the z-axis at index 75
                image_slice = image[:, :, 75]

                # Check the pixel type and convert to float32 if not supported
                if image_slice.GetPixelID() not in [sitk.sitkUInt8, sitk.sitkInt8, sitk.sitkUInt16, sitk.sitkInt16,
                                                    sitk.sitkFloat32]:
                    image_slice = sitk.Cast(image_slice, sitk.sitkFloat32)

                # Construct the output file path with ".tif" extension
                output_file_path = os.path.join(output_dir, file.replace(".nii.gz", ".tiff"))

                # Write the extracted slice to the output directory in TIFF format
                sitk.WriteImage(image_slice, output_file_path)

            except Exception as e:
                print(f"Error processing file {file}: {e}")
