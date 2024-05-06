import pathlib
import os

import imutils
import cv2

import webpdfparse

# Get the absolute path
CWD = pathlib.Path(os.path.abspath(__file__)).parent

results = webpdfparse.analyze_pdf(CWD / "behavior_mummy-1.pdf")

for id, page in enumerate(results.images):
    page_element = results.element.children[id]
    webpdfparse.draw_element(page, page_element, height=page.shape[0], width=page.shape[1])
    cv2.imshow(f"Page {id}", imutils.resize(page, width=700))

cv2.waitKey(0)
cv2.destroyAllWindows()