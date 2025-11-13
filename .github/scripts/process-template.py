#!/usr/bin/env python3
"""
Template processor for release reports using Python's built-in string.Template.
Much cleaner than bash/sed approach and no external dependencies!
"""

import os
import sys
import json
from datetime import datetime
from string import Template

def format_file_size(size_bytes):
    """
    Convert bytes to human-readable file size format
    """
    if size_bytes == 'unknown' or not size_bytes.isdigit():
        return size_bytes
    
    size_bytes = int(size_bytes)
    
    # Define size units
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    size = float(size_bytes)
    
    # Convert to appropriate unit
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    # Format with appropriate precision
    if unit_index == 0:  # Bytes
        return f"{int(size)} {units[unit_index]}"
    elif size >= 100:  # No decimal places for large numbers
        return f"{int(size)} {units[unit_index]}"
    else:  # One decimal place for smaller numbers
        return f"{size:.1f} {units[unit_index]}"

def generate_images_html(images, tag):
    """Generate HTML for the images section"""
    if not images:
        return "<p>No images available</p>"
    
    html_parts = []
    for i, image in enumerate(images):
        name = image.get('name', f'image-{i}')
        display_name = image.get('displayName', name.replace('-', ' ').title())
        url = image.get('url', '#')
        size = image.get('size', 'unknown')
        description = image.get('description', f'Download {display_name}')
        filename = image.get('filename', f'{name}-{tag}.tar.gz')
        
        # Format the file size for human readability
        formatted_size = format_file_size(size)
        
        html_parts.append(f'''
            <div class="image-card">
                <h3>{display_name}</h3>
                <p>{description}</p>
                <div class="download-info">
                    <span class="filename">{filename}</span>
                    <span class="size">{formatted_size}</span>
                </div>
                <a href="{url}" class="download-button">Download {display_name}</a>
            </div>
        ''')
    
    return ''.join(html_parts)

def parse_tab_delimited_images(images_data):
    """
    Parse tab-delimited image data into list of dictionaries
    
    DESIGN CHOICE: Tab-delimited instead of JSON
    ===========================================
    We use tab-delimited format instead of JSON because:
    1. AWS presigned URLs contain special characters (parentheses, ampersands, etc.)
       that cause shell escaping issues when passed as JSON command line arguments
    2. Presigned URLs NEVER contain actual tab characters (they use %09 encoding)
    3. Tab-delimited is much simpler - no JSON escaping/parsing needed
    4. More reliable - no shell escaping issues at all
    5. Faster processing - simple string splitting vs JSON parsing
    
    Format: name<TAB>displayName<TAB>filename<TAB>url<TAB>size<TAB>description
    """
    images = []
    for line in images_data.strip().split('\n'):
        if not line.strip():  # Skip empty lines
            continue
        parts = line.split('\t')
        if len(parts) != 6:
            print(f"❌ Error: Invalid tab-delimited line (expected 6 fields, got {len(parts)}): {line[:100]}...")
            sys.exit(1)
        
        images.append({
            'name': parts[0],
            'displayName': parts[1],
            'filename': parts[2],
            'url': parts[3],
            'size': parts[4],
            'description': parts[5]
        })
    return images

def main():
    if len(sys.argv) < 8:
        print("❌ Error: Not enough arguments provided")
        print("Usage: process-template.py <template_file> <output_file> <tag> <bucket> <tag_folder> <images_tsv_or_file> <repository> <run_id> [url_length] [expires_at]")
        print("Note: images_tsv_or_file can be either tab-delimited string or path to TSV file (prefixed with @)")
        sys.exit(1)
    
    template_file = sys.argv[1]
    output_file = sys.argv[2]
    tag = sys.argv[3]
    bucket = sys.argv[4]
    tag_folder = sys.argv[5]
    images_input = sys.argv[6]
    repository = sys.argv[7]
    run_id = sys.argv[8]
    url_length = sys.argv[9] if len(sys.argv) > 9 else 'unknown'
    expires_at = sys.argv[10] if len(sys.argv) > 10 else 'unknown'
    
    # Handle images input - either tab-delimited string or file path
    # We prefer file-based approach (@filename) to avoid shell escaping issues
    # when dealing with AWS presigned URLs containing special characters
    if images_input.startswith('@'):
        # Read from file (preferred approach)
        images_file = images_input[1:]  # Remove @ prefix
        if not os.path.exists(images_file):
            print(f"❌ Error: Images file not found: {images_file}")
            sys.exit(1)
        with open(images_file, 'r', encoding='utf-8') as f:
            images_data = f.read()
    else:
        # Use as tab-delimited string (fallback)
        images_data = images_input
    
    # Validate template file exists
    if not os.path.exists(template_file):
        print(f"❌ Error: Template file not found: {template_file}")
        sys.exit(1)
    
    # Parse images tab-delimited data
    try:
        images = parse_tab_delimited_images(images_data)
        print(f"📦 Processing {len(images)} images")
    except Exception as e:
        print(f"❌ Error: Invalid tab-delimited data: {e}")
        sys.exit(1)
    
    # Generate timestamp
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    
    print("🔄 Processing template with Python string.Template...")
    print(f"  Template: {template_file}")
    print(f"  Output: {output_file}")
    print(f"  Tag: {tag}")
    print(f"  Bucket: {bucket}")
    print(f"  Repository: {repository}")
    print(f"  Run ID: {run_id}")
    print()
    
    try:
        # Load template
        with open(template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # Convert {{variable}} syntax to $variable for string.Template
        # This is much simpler than using regex or complex string manipulation
        template_content = template_content.replace('{{TAG}}', '$TAG')
        template_content = template_content.replace('{{BUCKET}}', '$BUCKET')
        template_content = template_content.replace('{{TAG_FOLDER}}', '$TAG_FOLDER')
        template_content = template_content.replace('{{IMAGES}}', '$IMAGES')
        template_content = template_content.replace('{{REPOSITORY}}', '$REPOSITORY')
        template_content = template_content.replace('{{RUN_ID}}', '$RUN_ID')
        template_content = template_content.replace('{{TIMESTAMP}}', '$TIMESTAMP')
        template_content = template_content.replace('{{URL_LENGTH}}', '$URL_LENGTH')
        template_content = template_content.replace('{{EXPIRES_AT}}', '$EXPIRES_AT')
        
        # Create template object (using Python's built-in string.Template)
        template = Template(template_content)
        
        # Generate images HTML
        images_html = generate_images_html(images, tag)
        
        # Prepare context
        context = {
            'TAG': tag,
            'BUCKET': bucket,
            'TAG_FOLDER': tag_folder,
            'IMAGES': images_html,
            'REPOSITORY': repository,
            'RUN_ID': run_id,
            'TIMESTAMP': timestamp,
            'URL_LENGTH': url_length,
            'EXPIRES_AT': expires_at
        }
        
        # Render template (string.Template uses safe_substitute to avoid KeyError)
        rendered_content = template.safe_substitute(**context)
        
        # Write output
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(rendered_content)
        
        # Verify output
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            print("❌ Error: Failed to create output file or file is empty")
            sys.exit(1)
        
        # Check for unreplaced tokens
        if '{{' in rendered_content:
            print("⚠️  Warning: Some template tokens were not replaced")
            import re
            unreplaced = re.findall(r'\{\{[^}]+\}\}', rendered_content)
            for token in set(unreplaced):
                print(f"  - {token}")
        
        output_size = os.path.getsize(output_file)
        print("✅ Template processed successfully")
        print(f"  File: {output_file}")
        print(f"  Size: {output_size} bytes")
        
    except Exception as e:
        print(f"❌ Error processing template: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
