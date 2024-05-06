import pdfplumber
import cv2
import imutils
import pathlib
import pprint as pp
import os
import numpy as np
from typing import List

from .data_protocol import Element, ParseResults
from .vis import draw_element

CWD = pathlib.Path(os.path.abspath(__file__)).parent
GIT_ROOT = CWD.parent
EXAMPLES_DIR = GIT_ROOT / 'examples'

def coord_to_pixel(coord, page_width, page_height, img_width, img_height):
    x0 = (coord["x0"] / page_width) * img_width
    x1 = (coord["x1"] / page_width) * img_width
    y0 = (coord["top"] / page_height) * img_height
    y1 = (coord["bottom"] / page_height) * img_height
    return (int(x0), int(y0)), (int(x1), int(y1))

def identify_columns(rects, column_threshold):
    # Sort rects by their horizontal (x) position
    sorted_by_x = sorted(rects, key=lambda rect: rect['x'])

    columns = []
    current_column = [sorted_by_x[0]]

    for rect in sorted_by_x[1:]:
        last_rect = current_column[-1]

        # Check if the current rect is within the same column
        if abs(rect['x'] - last_rect['x']) < column_threshold:
            current_column.append(rect)
        else:
            columns.append(current_column)
            current_column = [rect]

    if current_column:
        columns.append(current_column)
    return columns

def group_images_into_image(rects, threshold):
    images = []

    for image in rects:
        if not images:
            images.append([image])
            continue

        for idx, group in enumerate(images):
            last_rect = group[-1]

            # Expand the rects via a margin
            last_rect_copy = last_rect.copy()
            last_rect_copy['x'] -= threshold
            last_rect_copy['y'] -= threshold
            last_rect_copy['width'] += threshold * 2
            last_rect_copy['height'] += threshold * 2

            # Image
            image_copy = image.copy()
            image_copy['x'] -= threshold
            image_copy['y'] -= threshold
            image_copy['width'] += threshold * 2
            image_copy['height'] += threshold * 2

            # Check if the current rect is within the same image via IoU
            iou = compute_iou(image_copy, last_rect_copy)
            if iou > 0:
                images[idx].append(image)
                break
        else:
            images.append([image])

    for image_group in images:
        x0 = min([rect['x'] for rect in image_group])
        y0 = min([rect['y'] for rect in image_group])
        x1 = max([rect['x'] + rect['width'] for rect in image_group])
        y1 = max([rect['y'] + rect['height'] for rect in image_group])
        image = {
            "x": x0,
            "y": y0,
            "width": x1 - x0,
            "height": y1 - y0
        }
        images[images.index(image_group)] = image

    return images

def group_lines_into_paragraphs(rects, column_threshold, vertical_threshold):
    paragraphs = []
    
    for column in identify_columns(rects, column_threshold):
        # Sort the rects in this column by their vertical (y) position
        sorted_by_y = sorted(column, key=lambda rect: rect['y'])
        current_paragraph = [sorted_by_y[0]]

        for rect in sorted_by_y[1:]:
            last_rect = current_paragraph[-1]

            # Check if the current rect is close enough vertically to be part of the same paragraph
            if rect['y'] - (last_rect['y'] + last_rect['height']) < vertical_threshold:
                current_paragraph.append(rect)
            else:
                paragraphs.append(current_paragraph)
                current_paragraph = [rect]

        if current_paragraph:
            paragraphs.append(current_paragraph)

    return paragraphs

def compute_iou(a_rect, b_rect):
    x_left = max(a_rect['x'], b_rect['x'])
    y_top = max(a_rect['y'], b_rect['y'])
    x_right = min(a_rect['x'] + a_rect['width'], b_rect['x'] + b_rect['width'])
    y_bottom = min(a_rect['y'] + a_rect['height'], b_rect['y'] + b_rect['height'])

    intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    area_a = a_rect['width'] * a_rect['height']
    area_b = b_rect['width'] * b_rect['height']
    iou = intersection / (area_a + area_b - intersection)

    return iou

def filter_paragraphs(paragraphs, image_rects):
    # Remove paragraphs that are more than 50% within an image
    filtered_paragraphs = []
    for idx, paragraph in enumerate(paragraphs):
        is_paragraph_within_image = False
        for image in image_rects:

            # Compute Union over Intersection (IoU) between the paragraph and the image
            iou = compute_iou(paragraph, image)

            if iou > 0.40:
                is_paragraph_within_image = True
                break

        for jdx, paragraph_a in enumerate(paragraphs):
            if idx == jdx:
                continue

            # Compute Union over Intersection (IoU) between the paragraph and another paragraph
            iou = compute_iou(paragraph, paragraph_a)
            if iou > 0.01:

                # Determine which paragraph is larger
                area_a = paragraph['width'] * paragraph['height']
                area_b = paragraph_a['width'] * paragraph_a['height']
                if area_a > area_b:
                    continue
                else:
                    is_paragraph_within_image = True
                    break

        if not is_paragraph_within_image:
            filtered_paragraphs.append(paragraph)

    return filtered_paragraphs

def process_pdf(idx, page, png) -> Element:
    # Extract the images from the page
    images_in_page = page.images
    page_height = page.height
    page_width = page.width
    image_rects = []

    for image in images_in_page:
        # Draw the image's rect on the page
        x = image['x0'] / page_width
        y = image['top'] / page_height
        width = (image['x1'] - image['x0']) / page_width
        height = (image['bottom'] - image['top']) / page_height
        # cv2.rectangle(pngs[idx], (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 5)
        image_rects.append({
            "x": x,
            "y": y,
            "width": width,
            "height": height
        })

    # Group the images into a single image
    image_rects = group_images_into_image(image_rects, threshold=0.1)
    image_elements = [
        Element(
            type='image',
            top_left=(rect['x'], rect['y']),
            bottom_right=(rect['x'] + rect['width'], rect['y'] + rect['height'])
        ) for rect in image_rects
    ]

    # Draw the image's rect on the page
    # for rect in image_rects:
    #     cv2.rectangle(png, (int(rect['x']), int(rect['y'])), (int(rect['x'] + rect['width']), int(rect['y'] + rect['height'])), (0, 0, 255), 5)

    # Extract the words from the page
    words = page.extract_words()
    words_rect = []
    for word in words:
        # (x0, y0), (x1, y1) = coord_to_pixel(word, page_width, page_height, png.shape[1], png.shape[0])
        # cv2.rectangle(png, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 5)
        words_rect.append({
            # "x": int(x0/page_width),
            # "y": int(y0/page_height),
            # "width": int((x1 - x0)/page_width),
            # "height": int((y1 - y0)/page_height),
            "x": word['x0']/page_width,
            "y": word['top']/page_height,
            "width": (word['x1'] - word['x0'])/page_width,
            "height": (word['bottom'] - word['top'])/page_height,
            'text': word['text']
        })

    # Extract the text from the page and draw the rect
    text = page.extract_text_lines(layout=True)
    line_rects = []
    for line in text:
        # (x0, y0), (x1, y1) = coord_to_pixel(line, page_width, page_height, png.shape[1], png.shape[0])
        # cv2.rectangle(png, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 5)
        line_rects.append({
            # "x": int(x0)/page_width,
            # "y": int(y0)/page_height,
            # "width": int(x1 - x0)/page_width,
            # "height": int(y1 - y0)/page_height,
            "x": line['x0']/page_width,
            "y": line['top']/page_height,
            "width": (line['x1'] - line['x0'])/page_width,
            "height": (line['bottom'] - line['top'])/page_height,
            "size": line['chars'][0]['size']
        })

    # Group the lines into paragraphs
    paragraphs_lines = group_lines_into_paragraphs(line_rects, column_threshold=0.1, vertical_threshold=0.01)
    paragraph_rects = []
    for paragraph in paragraphs_lines:
        # Draw a rect around the paragraph
        x0 = min([rect['x'] for rect in paragraph])
        y0 = min([rect['y'] for rect in paragraph])
        x1 = max([rect['x'] + rect['width'] for rect in paragraph])
        y1 = max([rect['y'] + rect['height'] for rect in paragraph])
        paragraph_rects.append({
            # "x": int(x0)/page_width,
            # "y": int(y0)/page_height,
            # "width": int(x1 - x0)/page_width,
            # "height": int(y1 - y0)/page_height,
            "x": x0,
            "y": y0,
            "width": x1 - x0,
            "height": y1 - y0,
            'lines': paragraph,
            'page': idx
        })

    # Filter out paragraphs that are entirely within an image or another paragraph
    filtered_paragraphs = filter_paragraphs(paragraph_rects, image_rects)

    paragraphs = []
    for p in filtered_paragraphs:
        paragraphs.append(Element(
            type='paragraph',
            top_left=(p['x'], p['y']),
            bottom_right=(p['x'] + p['width'], p['y'] + p['height']),
        )) 

    # Sort the paragraphs by their vertical (y) position and leftmost
    paragraphs = sorted(paragraphs, key=lambda p: (p.top_left[1], p.top_left[0]))

    # Create line elements
    lines = []
    for line in line_rects:
        lines.append(Element(
            type='line',
            top_left=(line['x'], line['y']),
            bottom_right=(line['x'] + line['width'], line['y'] + line['height']),
            meta_data={'size': line['size']}
        ))

    # Assign words to the lines
    for word in words_rect:
        # (x0, y0), (x1, y1) = coord_to_pixel(word, page_width, page_height, png.shape[1], png.shape[0])
        # centroid = ((x0 + x1) // 2, (y0 + y1) // 2)
        x0, y0, x1, y1 = word['x'], word['y'], word['x'] + word['width'], word['y'] + word['height']
        centroid = ((x0 + x1) / 2, (y0 + y1) / 2)
        for line in lines:
            if line.top_left[0] <= centroid[0] and line.top_left[1] <= centroid[1] and line.bottom_right[0] >= centroid[0] and line.bottom_right[1] >= centroid[1]:
                wordElement = Element(
                    type='word',
                    top_left=(x0, y0),
                    bottom_right=(x1, y1),
                    value=word['text'],
                    meta_data={'height': word['height']}
                )
                line.children.append(wordElement)

    # Sort the lines
    lines = sorted(lines, key=lambda l: (l.top_left[1], l.top_left[0]))
    
    # Sort words in the lines
    for line in lines:
        line.children = sorted(line.children, key=lambda w: (w.top_left[0], w.top_left[1]))

    # Assign lines to the paragraphs
    for line in lines:
        (x0, y0), (x1, y1) = line.top_left, line.bottom_right
        centroid = ((x0 + x1) / 2, (y0 + y1) / 2)
        for p in paragraphs:
            if p.top_left[0] <= centroid[0] and p.top_left[1] <= centroid[1] and p.bottom_right[0] >= centroid[0] and p.bottom_right[1] >= centroid[1]:
                p.children.append(line)

    # Remove outlier words (height bigger than median by 2 std)
    for p in paragraphs:
        heights = []
        for line in p.children:
            for word in line.children:
                heights.append(word.meta_data['height'])

        median = np.median(heights)
        std = np.std(heights)
        for line in p.children:
            line.children = [word for word in line.children if word.meta_data['height'] <= median + 2*std]

    # Set the end of lines to be the last word's end
    for p in paragraphs:
        for line in p.children:
            if line.children:
                line.bottom_right = line.children[-1].bottom_right

    # Determine the most common font size for each paragraph
    for p in paragraphs:
        font_sizes = [line.meta_data['size'] for line in p.children]
        font_size = np.median(font_sizes)
        p.meta_data['font_size'] = font_size

    return Element(
        type='page',
        top_left=(0, 0),
        bottom_right=(1, 1),
        # children=image_elements,
        children=image_elements+paragraphs,
        meta_data={'aspect-ratio': png.shape[1] / png.shape[0], 'page_number': idx}
    )

def subtyping_paragraphs(element: Element):

    font_sizes = [] 
    for page in element.children:
        for child in page.children:
            if child.type == 'paragraph':
                font_size = child.meta_data['font_size']
                font_sizes.append(font_size)

    if not font_sizes:
        return

    median_font_size = np.median(font_sizes)
    biggest_font_size = max(font_sizes)
    smallest_font_size = min(font_sizes)
    ratio = 0.2

    for page in element.children:
        for child in page.children:
            if child.type == 'paragraph':
                font_size = child.meta_data['font_size']
                if font_size == biggest_font_size:
                    child.type = "title"
                elif font_size >= median_font_size*(1+ratio):
                    child.type = "heading"
                elif font_size <= median_font_size*(1+ratio) and font_size >= median_font_size*(1-ratio):
                    child.type = "text"
            # elif font_size <= smallest_font_size*(1+ratio):
            #     child['type'] = "footer"
            # else:
            #     child['type'] = "unknown"
        
def analyze_pdf(path: pathlib.Path):
    
    # Open the PDF
    pdf = pdfplumber.open(str(path))
    # pngs = [cv2.imread(png) for png in pngs_files]
    pngs = [np.array(page.to_image(resolution=500).original) for page in pdf.pages]

    parent_element = Element(
        type='body',
        top_left=(0, 0),
        bottom_right=(pngs[0].shape[1], pngs[0].shape[0]),
        meta_data={'aspect-ratio': pngs[0].shape[1] / pngs[0].shape[0]}
    )
    children = []
    for idx, page in enumerate(pdf.pages):
        child = process_pdf(idx, page, pngs[idx])
        # draw_element(pngs[idx], child, height=pngs[idx].shape[0], width=pngs[idx].shape[1])
        children.append(child)

    parent_element.children = children

    # Typing the paragraphs
    # subtyping_paragraphs(parent_element)

    # Save to JSON
    # JSON_OUTPUT_DIR = DATASET_DIR / 'redforest' / 'pdfs' / 'json'
    # os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    # output_fp = JSON_OUTPUT_DIR / f'{path.stem}.json'
    # with open(output_fp, "w") as f:
    #     f.write(parent_element.to_json(indent=4))

    # return {'element': parent_element, 'images': pngs}
    return ParseResults(
        element=parent_element,
        images=pngs
    )