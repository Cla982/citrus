import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage import io
import shutil

# Configura il percorso base
base_path = "E:\\Tiff"  # Cambia questo percorso secondo le necessità
output_base_path = os.path.dirname(base_path)
processed_images_folder = os.path.join(output_base_path, 'Processed_Im')

# Percorso della cartella di origine
src_folder_path = base_path

# Percorso delle cartelle di output temporanee
jpg_folder = os.path.join(base_path, 'jpg')
vertical_sections_folder = os.path.join(base_path, 'VerticalSections')
otsu_folder = os.path.join(base_path, 'Otsu')

# Percorso della cartella di output finale
final_output_folder = processed_images_folder

# Assicurati che le cartelle di output esistano
if not os.path.exists(final_output_folder):
    os.makedirs(final_output_folder)

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
def process_image(file_path, output_path):
    image = Image.open(file_path).convert('L')
    img_array = np.array(image)

    # Applicare il limite di Otsu
    _, otsu_thresholded = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    final_image = Image.fromarray(otsu_thresholded.astype(np.uint8))
    final_image.save(output_path)

def process_folder(folder_path, output_folder):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, folder_path)
                output_directory = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)
                output_path = os.path.join(output_directory, file)
                process_image(file_path, output_path)

# Processa le immagini nella cartella jpg e salva in Otsu
process_folder(jpg_folder, otsu_folder)
print("Limite di Otsu applicato!")

# Funzione per caricare e ordinare le immagini
def load_images_from_folder(folder):
    image_files = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(folder) for f in filenames if f.endswith('.jpg') or f.endswith('.png')],
                         key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in image_files]
    subfolders = [os.path.relpath(os.path.dirname(file), folder) for file in image_files]
    return image_files, images, subfolders

def load_images(folder):
    images = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
    return images

# Funzione per creare sezioni verticali
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

# Funzione per ruotare le immagini
def rotate_image(image, angle=-90):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def correct_rotation(image):
    return rotate_image(image, 270)

# Processa le cartelle in Otsu e salva in VerticalSections
def process_folders(input_root, output_root):
    for subdir, dirs, files in os.walk(input_root):
        if files:
            input_folder = subdir
            output_folder = os.path.join(output_root, os.path.relpath(subdir, input_root))
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            images = load_images(input_folder)
            if images:
                vertical_sections = create_vertical_sections(images)
                for i, section in enumerate(vertical_sections):
                    cv2.imwrite(os.path.join(output_folder, f'vertical_section_{i*5}deg.jpg'), section)

process_folders(otsu_folder, vertical_sections_folder)
print("Sezione verticale eseguita!")

# Funzione per rimuovere righe di rumore nero
def remove_black_noise_lines(image, black_threshold=20):
    height, width = image.shape
    center_y = height // 2

    for y in range(center_y, -1, -1):
        if np.sum(image[y, :] < black_threshold) > 200:
            image[:y, :] = 0
            break

    for y in range(center_y, height):
        if np.sum(image[y, :] < black_threshold) > 200:
            image[y:, :] = 0
            break

    return image

# Funzione per calcolare e ridimensionare la bounding box del frutto
def calculate_and_resize_bounding_box(image):
    height, width = image.shape
    black_threshold = 50

    contours, _ = cv2.findContours((image > black_threshold).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    fruit = image[y:y+h, x:x+w]
    scale_factor = min(width / w, height / h)
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    fruit_resized = cv2.resize(fruit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    centered_image = np.zeros_like(image)
    start_x = (width - new_w) // 2
    start_y = (height - new_h) // 2
    centered_image[start_y:start_y+new_h, start_x:start_x+new_w] = fruit_resized

    return centered_image

# Funzione per oscurare le colonne centrali dell'immagine
def blackout_central_columns(image, center_column=128, column_width=25):
    height, width = image.shape
    start_col = center_column - column_width
    end_col = center_column + column_width
    image[:, start_col:end_col] = 0
    return image

# Funzione per processare le immagini
def process_image(image, output_folder, filename, subfolder):
    # Step 1: Remove black noise lines
    black_noise_removed_image = remove_black_noise_lines(image)

    # Step 2: Calculate and resize bounding box
    bounding_box_image = calculate_and_resize_bounding_box(black_noise_removed_image)

    # Step 3: Blackout central columns
    final_image = blackout_central_columns(bounding_box_image)

    # Save final image
    output_path = os.path.join(output_folder, subfolder)
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(os.path.join(output_path, filename), final_image)

def load_and_sort_images(folder_path):
    image_files = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(folder_path) for f in filenames if f.endswith('.jpg') or f.endswith('.png')],
                         key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in image_files]
    subfolders = [os.path.relpath(os.path.dirname(file), folder_path) for file in image_files]
    return image_files, images, subfolders

def process_images(input_folder, output_folder):
    image_files, images, subfolders = load_and_sort_images(input_folder)
    
    for idx, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        process_image(images[idx], output_folder, filename, subfolders[idx])

input_folder = vertical_sections_folder
output_folder = final_output_folder
process_images(input_folder, output_folder)
print("Immagini pronte!")

# Rimuovi le cartelle intermedie tranne 'jpg'
intermediate_dirs = [vertical_sections_folder, otsu_folder]

for dir_path in intermediate_dirs:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Removed directory: {dir_path}")

print("Cartelle intermedie rimosse!")
