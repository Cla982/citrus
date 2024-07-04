import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import shutil
from skimage import io

# Configura il percorso base
base_path = "L:\\Tiff"  # Cambia questo percorso secondo le necessità

# Percorso della cartella di origine
src_folder_path = base_path

# Percorso delle cartelle di output
jpg_folder = os.path.join(base_path, 'jpg')
vertical_sections_folder = os.path.join(base_path, 'VerticalSections')
otsu_folder = os.path.join(base_path, 'Otsu')

# Funzione per convertire e ridimensionare le immagini
def convert_and_resize_images(src_folder, dest_folder, size=(256, 256)):
    tiff_files = []
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
                tiff_files.append(os.path.join(root, file))
    
    for file_path in tqdm(tiff_files, desc="Converting images"):
        relative_path = os.path.relpath(os.path.dirname(file_path), src_folder)
        dest_path_root = os.path.join(dest_folder, relative_path)
        
        if not os.path.exists(dest_path_root):
            os.makedirs(dest_path_root)
        
        file_name = os.path.splitext(os.path.basename(file_path))[0] + '.jpg'
        dest_path_jpg = os.path.join(dest_path_root, file_name)
        
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        normalized_img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        resized_img = cv2.resize(normalized_img, size, interpolation=cv2.INTER_LANCZOS4)
        
        cv2.imwrite(dest_path_jpg, resized_img)

# Converti TIFF in JPG
convert_and_resize_images(src_folder_path, jpg_folder)
print("Conversione in JPG completata!")

# Funzione per applicare il limite di Otsu alle immagini
def apply_otsu_threshold(image):
    _, otsu_thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_thresholded

# Funzione per caricare e ordinare le immagini
def load_images_from_folder(folder):
    images = []
    filenames = sorted(os.listdir(folder))[0:924]
    for filename in filenames:
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def rotate_image(image, angle=-90):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def correct_rotation(image):
    return rotate_image(image, 270)

def create_vertical_sections(images, num_sections=72):
    sections = []
    for i in range(num_sections):
        angle = i * (360 / num_sections)
        rotated_images = [rotate_image(img, angle) for img in images]
        vertical_section = np.array([img[:, img.shape[1] // 2] for img in rotated_images]).T
        resized_section = cv2.resize(vertical_section, (256, 256), interpolation=cv2.INTER_AREA)
        corrected_section = correct_rotation(resized_section)
        sections.append(corrected_section)
    return sections

def process_folders(input_root, output_root):
    for subdir, dirs, files in os.walk(input_root):
        if files:
            input_folder = subdir
            output_folder = os.path.join(output_root, os.path.relpath(subdir, input_root))
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            images = load_images_from_folder(input_folder)
            if images:
                vertical_sections = create_vertical_sections(images)
                for i, section in enumerate(vertical_sections):
                    cv2.imwrite(os.path.join(output_folder, f'vertical_section_{i*5}deg.jpg'), section)

# Processa le cartelle in jpg e salva in VerticalSections
process_folders(jpg_folder, vertical_sections_folder)
print("Sezione verticale eseguita!")

# Funzione per rimuovere righe di rumore
def remove_noise_lines(image, threshold=50, white_threshold=200):
    height, width = image.shape
    black_threshold = 50  # Aggiungiamo un threshold per i pixel neri

    for y in range(height):
        row = image[y, :]
        left = np.argmax(row > white_threshold)
        right = width - np.argmax(np.flipud(row) > white_threshold)
        fruit_width = right - left

        if y > 0:
            prev_row = image[y-1, :]
            prev_left = np.argmax(prev_row > white_threshold)
            prev_right = width - np.argmax(np.flipud(prev_row) > white_threshold)
            prev_fruit_width = prev_right - prev_left

            if abs(fruit_width - prev_fruit_width) > threshold:
                # Identificare il rumore solo all'esterno del frutto
                for x in range(width):
                    if (x < left or x > right) and ((y > 0 and image[y-1, x] < black_threshold) or (y < height-1 and image[y+1, x] < black_threshold)):
                        image[y, x] = 0

    return image

def blackout_central_columns(image, center_column=128, column_width=25):
    height, width = image.shape
    start_col = center_column - column_width
    end_col = center_column + column_width
    image[:, start_col:end_col] = 0
    return image

def remove_vertical_lines(image, white_threshold=200, min_length=10):
    height, width = image.shape
    for x in range(width):
        col = image[:, x]
        if np.all(col > white_threshold):
            start_y = None
            for y in range(height):
                if col[y] <= white_threshold:
                    if start_y is not None and (y - start_y) >= min_length:
                        image[start_y:y, x] = 0
                    start_y = None
                elif start_y is None:
                    start_y = y
    return image

def process_image(image):
    # Applica il limite di Otsu
    image = apply_otsu_threshold(image)

    height, width = image.shape
    center_y = height // 2

    # Define threshold for considering a pixel as black
    black_threshold = 20

    # Scroll upwards from the center and black out above the first row with more than 200 black pixels
    for y in range(center_y, -1, -1):
        black_pixel_count = np.sum(image[y, :] < black_threshold)
        if black_pixel_count > 200:
            image[:y, :] = 0
            break
    
    # Scroll downwards from the center and black out below the first row with more than 200 black pixels
    for y in range(center_y, height):
        black_pixel_count = np.sum(image[y, :] < black_threshold)
        if black_pixel_count > 200:
            image[y:, :] = 0
            break

    # Remove noise lines
    image = remove_noise_lines(image)

    # Remove vertical lines
    image = remove_vertical_lines(image)

    # Calculate the bounding box of the fruit
    contours, _ = cv2.findContours((image > black_threshold).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Extract the fruit and calculate the scaling factor
    fruit = image[y:y+h, x:x+w]
    scale_factor = min(width / w, height / h)
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    # Resize the fruit while maintaining the aspect ratio
    fruit_resized = cv2.resize(fruit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a new black image and place the resized fruit at the center
    centered_image = np.zeros_like(image)
    start_x = (width - new_w) // 2
    start_y = (height - new_h) // 2
    centered_image[start_y:start_y+new_h, start_x:start_x+new_w] = fruit_resized
    
    # Blackout central columns
    centered_image = blackout_central_columns(centered_image)

    return centered_image

def load_and_sort_images(folder_path):
    image_files = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(folder_path) for f in filenames if f.endswith('.jpg') or f.endswith('.png')],
                         key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in image_files]
    return image_files, images

def process_images(input_folder, output_folder):
    image_files, images = load_and_sort_images(input_folder)
    
    for idx, img_path in enumerate(image_files):
        relative_path = os.path.relpath(img_path, input_folder)
        output_path = os.path.join(output_folder, relative_path)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        processed_img = process_image(images[idx])
        io.imsave(output_path, processed_img)

input_folder = vertical_sections_folder
output_folder = os.path.join(base_path, 'Processed_Images')
process_images(input_folder, output_folder)
print("Immagini pronte!")

# Rimuovi le cartelle intermedie
intermediate_dirs = [jpg_folder, vertical_sections_folder, otsu_folder]

for dir_path in intermediate_dirs:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Removed directory: {dir_path}")

print("Cartelle intermedie rimosse!")
