from PIL import Image
import glob
import os

# ---- EDIT THIS to your VMD working directory ----
folder = r"C:\Users\erenr\Documents\McGill\Research\Chem396-Amorphous-Silicon\VMD"

files = sorted(glob.glob(os.path.join(folder, "*.bmp")))
gif_path = os.path.join(folder, "melt_216_22almtp_2500K.gif")

if not files:
    print("No .bmp files found — check the folder path.")
else:
    print(f"Found {len(files)} frames, building GIF...")
    frames = [Image.open(f) for f in files]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0
    )
    # close the file handles so Windows lets us delete them
    for im in frames:
        im.close()

    # only delete if the GIF exists and isn't empty
    if os.path.exists(gif_path) and os.path.getsize(gif_path) > 0:
        for f in files:
            os.remove(f)
        print(f"Done: GIF created and {len(files)} .bmp frames deleted.")
    else:
        print("GIF was not created properly — frames left untouched.")