import os
import nibabel as nib
import matplotlib.pyplot as plt

class FourPaneViewer:
    def __init__(self, patient_dir):
        # Modalities in display order
        # t1 = T1-weighted (highlights fat-rich tissues),
        # t1ce = T1 with contrast (highlights tumors & inflammation),
        # t2 = T2-weighted (highlights fluid filled tissues),
        # flair = FLAIR (Fluid Attenuated Inversion Recovery, similar to T2 but suppresses CSF - cerebrospinal fluid)
        self.modalities = ["t1", "t1ce", "t2", "flair"]

        # Load each modality
        self.imgs = {}
        for mode in self.modalities:
            file = [file for file in os.listdir(patient_dir) if mode in file.lower() and file.endswith(".nii.gz")]
            if not file:
                raise FileNotFoundError(f"Missing modality {mode}")
            self.imgs[mode] = nib.load(os.path.join(patient_dir, file[0])).get_fdata()
            print('{} shape: {}'.format(mode, self.imgs[mode].shape))

        # Load segmentation mask (outlines the tumor we're interested in - what we want to eventuall predict)
        seg_file = [file for file in os.listdir(patient_dir) if "seg" in file.lower() and file.endswith(".nii.gz")]
        if not seg_file:
            raise FileNotFoundError("Missing segmentation mask")
        self.mask = nib.load(os.path.join(patient_dir, seg_file[0])).get_fdata()

        # Initial slice index - start at the middle slice 
        self.slice_idx = self.imgs[self.modalities[0]].shape[2] // 2 # shape[2] = z-axis

        # Matplotlib setup (2x2 grid)
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 10))
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.update_display()
        # Save the modalities figure
        plt.savefig(os.path.join('images', 'orig_all_modalities.png'), bbox_inches='tight', dpi=300)
        plt.show()

    def update_display(self):
        # We loop through each subplot axis and modality to update the display
        # .flat - flattens the 2D array of axes to a 1D iterable so we go through the subplot axes in order: 
        # top-left, top-right, bottom-left, bottom-right
        # zip combines the axes with the modalities so that in each iteration the axis corresponds to the right modality
        for axis, modality in zip(self.axes.flat, self.modalities):
            # clear the axis for fresh display
            axis.clear()

            # We grab all x & y slices (2D image) for the current z slice
            img_slice = self.imgs[modality][:, :, self.slice_idx]
            mask_slice = self.mask[:, :, self.slice_idx]

            # Display the full image slice and overlay the segmentation mask in lower opacity
            axis.imshow(img_slice.T, cmap='gray', origin='lower')
            axis.imshow(mask_slice.T, cmap='jet', alpha=0.4, origin='lower')
            axis.set_title(f"{modality.upper()} | Slice {self.slice_idx}")
            axis.axis('off')

        self.fig.suptitle("BraTS Multi-Modal Viewer", fontsize=16)

        # Efficiently update the display
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        if event.button == 'up':
            # min ensures we don't go beyond the last slice
            self.slice_idx = min(self.slice_idx + 1, self.mask.shape[2] - 1) # shape[2] = z-axis
        elif event.button == 'down':
            # max ensures we don't go below the first slice
            self.slice_idx = max(self.slice_idx - 1, 0)
        self.update_display()

if __name__ == "__main__":
    # Example folder: data/BraTS21_Training_001
    patient_dir = "data\\BraTS2021_Training_Data\\BraTS2021_00000"
    FourPaneViewer(patient_dir)
