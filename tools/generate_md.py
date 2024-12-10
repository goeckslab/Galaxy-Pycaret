import base64
import logging
import os

from markdown2 import markdown

import pandas as pd

from weasyprint import HTML

LOG = logging.getLogger(__name__)


def dataframe_to_markdown(df):
    # Generate header
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = "\n".join(["| " + " | ".join(map(str, row)) + " |" for row in df.values])

    return f"{header}\n{separator}\n{rows}"

def generate_report_from_path(base_path, output_path, format="md"):
    """
    Generate a report (Markdown or PDF) based on the directory structure.

    Parameters:
        base_path (str): Path to the base directory containing subfolders.
        output_path (str): Path to save the generated report.
        format (str): Output format ("md" for Markdown, "pdf" for PDF).

    Returns:
        str: Path to the saved report.
    """
    LOG.info(f"Generating report from path: {base_path} as {format.upper()}")

    # Initialize the report content
    markdown_content = "# Report\n\n---\n"

    # Iterate through each subfolder in the base path
    for folder_name in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue  # Skip files, only process directories

        markdown_content += f"## {folder_name.capitalize()}\n\n"

        # Process CSV files in the subfolder
        for file_name in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, file_name)
            if file_name.lower().endswith(".csv"):
                try:
                    markdown_content += f"### Table: {file_name}\n\n"
                    table = pd.read_csv(file_path)
                    markdown_content += dataframe_to_markdown(table)
                    markdown_content += "\n\n"
                except Exception as e:
                    LOG.error(f"Failed to process CSV {file_path}: {e}")
                    markdown_content += f"*Error reading table: {file_name}*\n\n"

        # Process image files in the subfolder
        for file_name in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, file_name)
            if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    encoded_image = encode_image_to_base64(file_path)
                    image_name = os.path.splitext(file_name)[0]
                    markdown_content += f"### Plot: {image_name}\n\n"
                    if format == "md":
                        markdown_content += f"![{image_name}](data:image/png;base64,{encoded_image})\n\n"
                    else:
                        # Include image as file path for PDF rendering
                        markdown_content += (
                            f'<img src="data:image/png;base64,{encoded_image}" alt="{file_name}"'
                            f'style="width:600px; max-width:100%; height:auto;" />\n\n'
                        )
                        LOG.error(f"Image path: {file_path}")
                except Exception as e:
                    LOG.error(f"Failed to process image {file_path}: {e}")
                    markdown_content += f"*Error displaying plot: {file_name}*\n\n"

        markdown_content += "---\n"

    # Save to Markdown or PDF based on the format
    if format == "md":
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as file:
            file.write(markdown_content)
        LOG.info(f"Markdown report saved at: {output_path}")
    elif format == "pdf":
        html_content = markdown(markdown_content, extras=["tables"])
        HTML(string=html_content).write_pdf(output_path)
        LOG.error(f"PDF report saved at: {output_path}")
        return output_path

    return output_path


def encode_image_to_base64(img_path):
    """
    Encode an image to Base64 format.

    Parameters:
        img_path (str): Path to the image file.

    Returns:
        str: Base64-encoded string of the image.
    """
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
