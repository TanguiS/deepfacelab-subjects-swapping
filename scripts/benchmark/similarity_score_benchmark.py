import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm


def flip_image(image_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    flipped_image = cv2.flip(image, 1)  # Flip horizontally
    cv2.imwrite(str(output_path), flipped_image)


def rotate_image(image_path: Path, output_path: Path, angle: float) -> None:
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    cv2.imwrite(str(output_path), rotated_image)


def scale_image(image_path: Path, output_path: Path, scale_factor: float) -> None:
    image = cv2.imread(str(image_path))
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    cv2.imwrite(str(output_path), scaled_image)


def translate_image(image_path: Path, output_path: Path, tx: int, ty: int) -> None:
    image = cv2.imread(str(image_path))
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    cv2.imwrite(str(output_path), translated_image)


def color_jitter_image(image_path: Path, output_path: Path, brightness: int, contrast: float,
                       saturation: float) -> None:
    image = cv2.imread(str(image_path))
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv[..., 1] = image_hsv[..., 1] * saturation
    image_jittered = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(output_path), image_jittered)


def add_noise_to_image(image_path: Path, output_path: Path, mean: float, std_dev: float) -> None:
    image = cv2.imread(str(image_path))
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    cv2.imwrite(str(output_path), noisy_image)


def crop_image(image_path: Path, output_path: Path, x_percent: float, y_percent: float, width_percent: float, height_percent: float) -> None:
    image = cv2.imread(str(image_path))
    height, width, _ = image.shape
    x = int(width * x_percent)
    y = int(height * y_percent)
    width = int(width * width_percent)
    height = int(height * height_percent)
    cropped_image = image[y:y + height, x:x + width]
    cv2.imwrite(str(output_path), cropped_image)


def add_padding_to_image(image_path: Path, output_path: Path, top: int, bottom: int, left: int, right: int) -> None:
    image = cv2.imread(str(image_path))
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    cv2.imwrite(str(output_path), padded_image)


def output_file_name(input_path: Path, index: int, method: str) -> Path:
    parent_dir = input_path.parent
    new_path = parent_dir.joinpath(f"{input_path.stem}_{index}_{method}{input_path.suffix}")
    return new_path


def apply_random_data_augmentation(image_path: Path, index: int, method: str) -> None:
    if method == "flip":
        flip = random.choice([True, False])
        flip_image(image_path, output_file_name(image_path, index, "flip")) if flip else None
    elif method == "rotate":
        rotation_angle = random.uniform(-25, 25)
        rotate_image(image_path, output_file_name(image_path, index, "rotate"), rotation_angle)
    elif method == "scale":
        scale_factor = random.uniform(0.02, 1.8)
        scale_image(image_path, output_file_name(image_path, index, "scale"), scale_factor)
    elif method == "translate":
        translation_x = random.randint(-100, 100)
        translation_y = random.randint(-100, 100)
        translate_image(image_path, output_file_name(image_path, index, "translate"), translation_x, translation_y)
    elif method == "color_jitter":
        brightness_factor = random.uniform(0.1, 1.8)
        contrast_factor = random.uniform(0.1, 1.8)
        saturation = random.uniform(0.35, 1.5)
        color_jitter_image(image_path, output_file_name(image_path, index, "color_jitter"), brightness_factor,
                           contrast_factor, saturation)
    elif method == "noise":
        noise_mean = random.uniform(0, 150)
        noise_stddev = random.uniform(150, 225)
        add_noise_to_image(image_path, output_file_name(image_path, index, "noise"), noise_mean, noise_stddev)
    elif method == "crop":
        crop_x_percent = random.uniform(0.05, 0.1)
        crop_y_percent = random.uniform(0.05, 0.1)
        crop_width_percent = random.uniform(0.1, 0.3)
        crop_height_percent = random.uniform(0.1, 0.3)
        crop_image(image_path, output_file_name(image_path, index, "crop"), crop_x_percent, crop_y_percent, crop_width_percent, crop_height_percent)
    elif method == "padding":
        padding_left = random.randint(10, 30)
        padding_right = random.randint(10, 30)
        padding_top = random.randint(10, 30)
        padding_bottom = random.randint(10, 30)
        add_padding_to_image(image_path, output_file_name(image_path, index, "padding"), padding_left, padding_right,
                             padding_top, padding_bottom)


def launch(benchmark_dir: Path, number_data_augmentation_loop: int):
    images = [img for img in benchmark_dir.iterdir()]
    data_set = images.copy()
    methods = {"scale", "color_jitter", "noise"}
    for _ in tqdm(range(0, number_data_augmentation_loop), desc="Loading images"):
        data_set.extend(images)

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = []
        for i, image in enumerate(data_set):
            for method in methods:
                futures.append(executor.submit(apply_random_data_augmentation, image, i, method))
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Augmentation", unit="image"):
            pass

