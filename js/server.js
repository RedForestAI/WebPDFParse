const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { findAOIs } = require('./aoiFinder'); // Handle PDF conversion and drawing

const app = express();
const PORT = 3001;
const upload = multer({ dest: 'uploads/' });

app.use(express.static('public'));

app.post('/process-pdf', upload.single('pdf'), async (req, res) => {
    if (!req.file) {
        return res.status(400).send('No PDF file uploaded.');
    }

    try {
        // findAOIs will handle PDF conversion and image processing
        const resultPath = await findAOIs(req.file.path);
        res.sendFile(resultPath);
    } catch (error) {
        console.error(error);
        res.status(500).send('Error processing PDF.');
    } finally {
        // Clean up uploaded PDF
        fs.unlinkSync(req.file.path);
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
