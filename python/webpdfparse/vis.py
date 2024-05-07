from typing import List

import numpy as np
import cv2

from .data_protocol import Element

colormap = {
    "body": (0, 0, 0),
    "page": (0, 255, 0),
    "block": (0, 0, 255),
    "paragraph": (0, 0, 255),
    "image": (0, 0, 255),
    "heading": (0, 255, 255),
    "line": (255, 0, 0),
    "word": (0, 255, 0),
    # 'title': (255, 0, 255)
}

# First, let's identify the paragraphs
def draw_element(
        draw_img: np.ndarray, 
        element: Element, 
        height: int, 
        width: int,
        subtype: List[str] = None    
    ):

    # if element.type in ['page', 'paragraph', 'quote', 'image', 'caption', 'line', 'word']:
    if type(subtype) == type(None) or element.type in subtype:

        center_x = element.top_left[0] + (element.bottom_right[0] - element.top_left[0]) // 2
        center_y = element.top_left[1] + (element.bottom_right[1] - element.top_left[1]) // 2

        # Convert x and y to absolute coordinates
        center_x = int(center_x * width)
        center_y = int(center_y * height)
        top_left = (int(element.top_left[0]*width), int(element.top_left[1]*height))
        bottom_right = (int(element.bottom_right[0]*width), int(element.bottom_right[1]*height))

        color = colormap[element.type]
        center = (center_x-len(element.address)*30, center_y)
        cv2.putText(draw_img, str(element.address), center, cv2.FONT_HERSHEY_SIMPLEX, 5, color, 5)
        cv2.rectangle(draw_img, top_left, bottom_right, color, 10)

    for i,e in enumerate(element.children):
        draw_element(draw_img, e, height, width, subtype)