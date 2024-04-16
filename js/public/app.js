document.getElementById('file-input').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file && file.type === 'application/pdf') {
        // Show the view PDF button
        document.getElementById('view-pdf-btn').style.display = 'inline-block';
    } else {
        // Hide the view PDF button if no file is selected or file is not a PDF
        document.getElementById('view-pdf-btn').style.display = 'none';
    }
});

document.getElementById('view-pdf-btn').addEventListener('click', function() {
    const fileInput = document.getElementById('file-input');
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        const fileReader = new FileReader();
        fileReader.onload = function() {
            const pdfData = fileReader.result;
            const viewer = document.getElementById('pdf-viewer');
            viewer.src = pdfData;
            viewer.hidden = false;
        };
        fileReader.readAsDataURL(file);
    }
});
