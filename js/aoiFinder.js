const { convert } = require('pdf-poppler');
const path = require('path');
const fs = require('fs');
// const cv = require('opencv4nodejs');

async function findAOIs(pdfPath) {
    let opts = {
        format: 'jpeg',
        out_dir: path.dirname(pdfPath),
        out_prefix: path.basename(pdfPath, path.extname(pdfPath)),
        page: null // convert all pages
    };

    try {
        await convert(pdfPath, opts);
        const images = fs.readdirSync(opts.out_dir).filter(f => f.includes(opts.out_prefix));

        // Process each image to find and mark areas of interest
        images.forEach(image => {
            //process images
        });

        // Return the path to the processed images or a combined image/pdf
        return path.join(opts.out_dir, images[0]); 
    } catch (error) {
        console.error('Error converting PDF:', error);
        throw error;
    }
}

module.exports = { findAOIs };
