import pathlib
import os
import json

import imutils
import cv2

import webpdfparse

# Get the absolute path
CWD = pathlib.Path(os.path.abspath(__file__)).parent

results = webpdfparse.analyze_pdf(CWD / "behavior_mummy-1.pdf")

# Save as a JSON
output_fp = CWD / 'behavior_example.json'
with open(output_fp, 'w') as f:
    f.write(json.dumps(results.element.to_dict(), indent=4))

for id, page in enumerate(results.images):
    page_element = results.element.children[id]
    webpdfparse.draw_element(page, page_element, height=page.shape[0], width=page.shape[1])
    # cv2.imshow(f"Page {id}", imutils.resize(page, width=700))
    cv2.imwrite(str(CWD / f"page_{id}.png"), page)


cv2.waitKey(0)
cv2.destroyAllWindows()