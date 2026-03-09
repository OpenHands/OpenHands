# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
"""Utility module for generating file viewer HTML content."""

import base64
import mimetypes
import os


def generate_file_viewer_html(file_path: str) -> str:
    """Generate HTML content for viewing different file types.

    Args:
        file_path: The absolute path to the file

    Returns:
        str: HTML content for viewing the file

    Raises:
        ValueError: If the file extension is not supported
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)

    # Define supported file extensions
    supported_extensions = [
        '.pdf',
        '.png',
        '.jpg',
        '.jpeg',
        '.gif',
    ]

    # Check if the file extension is supported
    if file_extension not in supported_extensions:
        raise ValueError(
            f'Unsupported file extension: {file_extension}. '
            f'Supported extensions are: {", ".join(supported_extensions)}'
        )

    # Check if the file exists
    if not os.path.exists(file_path):
        raise ValueError(
            f'File not found locally: {file_path}. Please download the file to the local machine and try again.'
        )

    # Read file content directly
    file_content = None
    mime_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'

    # For binary files (images, PDFs), encode as base64
    if file_extension in ['.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp']:
        with open(file_path, 'rb') as file:
            file_content = base64.b64encode(file.read()).decode('utf-8')
    # For text files, read as text
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Viewer - {file_name}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
    <style>
        body, html {{ margin: 0; padding: 0; height: 100%; overflow: hidden; font-family: Arial, sans-serif; }}
        #viewer-container {{ width: 100%; height: 100vh; overflow: auto; }}
        .page {{ margin: 10px auto; box-shadow: 0 0 10px rgba(0,0,0,0.3); }}
        .text-content {{ margin: 20px; white-space: pre-wrap; font-family: monospace; line-height: 1.5; }}
        .error {{ color: red; margin: 20px; }}
        img {{ max-width: 100%; margin: 20px auto; display: block; }}
    </style>
</head>
<body>
    <div id="viewer-container"></div>
    <script>
    const filePath = "{file_path}";
    const fileExtension = "{file_extension}";
    const fileContent = `{file_content if file_extension not in ['.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp'] else ''}`;
    const fileBase64 = "{file_content if file_extension in ['.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp'] else ''}";
    const mimeType = "{mime_type}";
    const container = document.getElementById('viewer-container');

    async function loadContent() {{
        try {{
            if (fileExtension === '.pdf') {{
                pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
                const binaryString = atob(fileBase64);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {{
                    bytes[i] = binaryString.charCodeAt(i);
                }}

                const loadingTask = pdfjsLib.getDocument({{data: bytes.buffer}});
                const pdf = await loadingTask.promise;

                // Get total number of pages
                const numPages = pdf.numPages;

                // Render each page
                for (let pageNum = 1; pageNum <= numPages; pageNum++) {{
                    const page = await pdf.getPage(pageNum);

                    // Set scale for rendering
                    const viewport = page.getViewport({{ scale: 1.5 }});

                    // Create canvas for rendering
                    const canvas = document.createElement('canvas');
                    canvas.className = 'page';
                    canvas.width = viewport.width;
                    canvas.height = viewport.height;
                    container.appendChild(canvas);

                    // Render PDF page into canvas context
                    const context = canvas.getContext('2d');
                    const renderContext = {{
                        canvasContext: context,
                        viewport: viewport
                    }};

                    await page.render(renderContext).promise;
                }}
            }} else if (['.png', '.jpg', '.jpeg', '.gif', '.bmp'].includes(fileExtension)) {{
                const img = document.createElement('img');
                img.src = `data:${{mimeType}};base64,${{fileBase64}}`;
                img.alt = filePath.split('/').pop();
                container.appendChild(img);
            }} else {{
                const pre = document.createElement('pre');
                pre.className = 'text-content';
                pre.textContent = fileContent;
                container.appendChild(pre);
            }}
        }} catch (error) {{
            console.error('Error:', error);
            container.innerHTML = `<div class="error"><h2>Error loading file</h2><p>${{error.message}}</p></div>`;
        }}
    }}

    window.onload = loadContent;
    </script>
</body>
</html>"""
