// Assuming PDF.js is already included and set up in your environment

// Load the PDF file
pdfjsLib.getDocument('path/to/your/document.pdf').promise.then(function(pdfDoc) {
    // Get a specific page (e.g., first page)
    pdfDoc.getPage(1).then(function(page) {
        // You can now access page methods and properties
        extractTextData(page);
    });
});

function extractTextData(page) {
    // Get text content
    page.getTextContent().then(function(textContent) {
        textContent.items.forEach(function(item) {
            const transform = item.transform;

            // Calculate position and dimensions
            const x = transform[4];
            const y = transform[5];
            const width = item.width;
            const height = item.height;

            console.log(`Text: ${item.str}, Position: (${x}, ${y}), Size: (${width} x ${height})`);
        });
    });
}
