from loguru import logger
from pathlib import Path
import pytesseract
from PIL import Image

def ocr_vi_fr(image_path: Path, output_path: Path) -> None:
    """
    Placeholder function for OCR processing.
    This function should implement the OCR logic for Vietnamese and French text.

    Args:
        image_path (Path): Path to the input image file.
        output_path (Path): Path to save the OCR output text file.
    """
    if not image_path.exists():
        logger.error(f"Image file does not exist: {image_path}")
        return

    try:
        # Open the image file
        with Image.open(image_path) as img:
            # Perform OCR using pytesseract
            text = pytesseract.image_to_string(img, lang='vie+fra')

        # Save the extracted text to the output file
        output_path.write_text(text, encoding='utf-8')
        logger.success(f"OCR output saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error during OCR processing: {e}")

def ocrize_images_in_folder(folder_path: Path, output_folder: Path, extension: str = "jpg") -> None:
    """
    Process all images in a folder and save OCR results to text files.

    Args:
        folder_path (Path): Path to the folder containing images.
        output_folder (Path): Path to save the OCR output text files.
        extension (str): File extension of images to process (default is 'jpg').
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    images = sorted(folder_path.glob(f'*.{extension}'))  # Adjust the glob pattern as needed
    if not images:
        logger.warning(f"No images with extension '{extension}' found in {folder_path}.")
        return

    for image_file in images:  # Adjust the glob pattern as needed
        output_file = output_folder / f"{image_file.stem}.txt"
        ocr_vi_fr(image_file, output_file)


if __name__ == "__main__":
    input_folder = Path("data/Nam-Phong/Quyen-1/So-1/gold")
    output_folder = Path("data/Nam-Phong/Quyen-1/So-1/gold")

    ocrize_images_in_folder(input_folder, output_folder)
    logger.info("OCR processing completed.")
