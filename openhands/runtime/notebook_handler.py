"""
OpenHands FileEditor Extension for .ipynb (Jupyter Notebook) Support

This module extends OHEditor to handle Jupyter notebooks specially to:
1. Prevent LLM context overflow from large outputs
2. Provide clean, readable notebook representations
3. Extract only relevant code and markdown content
"""

import json
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass


@dataclass
class NotebookCell:
    """Represents a simplified notebook cell."""
    cell_type: str  # 'code', 'markdown', 'raw'
    source: str
    execution_count: Optional[int] = None
    output_summary: Optional[str] = None
    has_image_output: bool = False


class NotebookHandler:
    """
    Handler for Jupyter notebook (.ipynb) files.
    
    Prevents LLM context overflow by:
    - Truncating large cell outputs
    - Converting base64 images to placeholders
    - Limiting total content size
    """
    
    # Size limits to prevent context overflow
    MAX_OUTPUT_CHARS = 500  # Max characters per cell output
    MAX_TOTAL_CELLS = 100   # Max cells to process
    MAX_SOURCE_LINES = 200  # Max lines per cell source
    
    # Image MIME types that will be replaced with placeholders
    IMAGE_MIME_TYPES = {
        'image/png', 'image/jpeg', 'image/jpg', 'image/gif', 
        'image/svg+xml', 'image/webp'
    }
    
    def __init__(self, 
                 max_output_chars: int = 500,
                 max_total_cells: int = 100,
                 max_source_lines: int = 200):
        self.max_output_chars = max_output_chars
        self.max_total_cells = max_total_cells
        self.max_source_lines = max_source_lines
    
    def is_notebook(self, path: Path) -> bool:
        """Check if file is a Jupyter notebook."""
        return path.suffix.lower() == '.ipynb'
    
    def process_notebook(self, path: Path) -> str:
        """
        Process a notebook file and return a clean, LLM-friendly representation.
        
        Returns a formatted string with:
        - Notebook metadata
        - Cell contents (code/markdown)
        - Truncated/summarized outputs
        - Image placeholders
        """
        with open(path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        cells = self._extract_cells(notebook)
        return self._format_notebook(cells, notebook.get('metadata', {}))
    
    def _extract_cells(self, notebook: Dict[str, Any]) -> List[NotebookCell]:
        """Extract and process cells from notebook JSON."""
        cells = []
        raw_cells = notebook.get('cells', [])
        
        # Limit total cells to prevent overflow
        for i, cell in enumerate(raw_cells[:self.max_total_cells]):
            cell_type = cell.get('cell_type', 'code')
            
            # Process source
            source = self._process_source(cell.get('source', []))
            
            # Process outputs for code cells
            output_summary = None
            has_image = False
            
            if cell_type == 'code' and 'outputs' in cell:
                output_summary, has_image = self._process_outputs(cell['outputs'])
            
            cells.append(NotebookCell(
                cell_type=cell_type,
                source=source,
                execution_count=cell.get('execution_count'),
                output_summary=output_summary,
                has_image_output=has_image
            ))
        
        # Add truncation notice if needed
        if len(raw_cells) > self.max_total_cells:
            cells.append(NotebookCell(
                cell_type='markdown',
                source=f'\n[... {len(raw_cells) - self.max_total_cells} more cells truncated ...]\n',
                output_summary=None
            ))
        
        return cells
    
    def _process_source(self, source: Union[str, List[str]]) -> str:
        """Process cell source, limiting lines if necessary."""
        if isinstance(source, list):
            source = ''.join(source)
        
        lines = source.split('\n')
        if len(lines) > self.max_source_lines:
            lines = lines[:self.max_source_lines]
            lines.append(f'\n[... {len(lines) - self.max_source_lines} more lines truncated ...]')
        
        return '\n'.join(lines)
    
    def _process_outputs(self, outputs: List[Dict]) -> tuple[Optional[str], bool]:
        """
        Process cell outputs, truncating and summarizing.
        
        Returns: (output_summary, has_image)
        """
        if not outputs:
            return None, False
        
        processed_outputs = []
        has_image = False
        
        for output in outputs:
            output_type = output.get('output_type', '')
            
            # Handle stream outputs (stdout/stderr)
            if output_type in ('stream',):
                text = output.get('text', [])
                if isinstance(text, list):
                    text = ''.join(text)
                processed_outputs.append(self._truncate_text(text, prefix='[stdout] '))
            
            # Handle execute_result and display_data
            elif output_type in ('execute_result', 'display_data'):
                data = output.get('data', {})
                
                # Check for images
                for mime_type in self.IMAGE_MIME_TYPES:
                    if mime_type in data:
                        has_image = True
                        size_info = self._get_base64_size(data[mime_type])
                        processed_outputs.append(f'[Image: {mime_type}, {size_info}]')
                        break
                else:
                    # Text output
                    if 'text/plain' in data:
                        text = data['text/plain']
                        if isinstance(text, list):
                            text = ''.join(text)
                        processed_outputs.append(self._truncate_text(text))
                    elif 'text/html' in data:
                        processed_outputs.append('[HTML output - truncated]')
                    elif 'application/json' in data:
                        processed_outputs.append('[JSON output - truncated]')
            
            # Handle error outputs
            elif output_type == 'error':
                ename = output.get('ename', 'Error')
                evalue = output.get('evalue', '')
                processed_outputs.append(f'[Error: {ename}: {self._truncate_text(evalue)}]')
        
        summary = '\n'.join(processed_outputs) if processed_outputs else None
        return summary, has_image
    
    def _truncate_text(self, text: str, prefix: str = '') -> str:
        """Truncate text to max length."""
        if len(text) > self.max_output_chars:
            return prefix + text[:self.max_output_chars] + f'\n[... {len(text) - self.max_output_chars} more chars truncated ...]'
        return prefix + text
    
    def _get_base64_size(self, data: Union[str, List[str]]) -> str:
        """Estimate size of base64 image data."""
        if isinstance(data, list):
            data = ''.join(data)
        # Base64 is ~4/3 of binary size
        binary_size = len(data) * 0.75
        if binary_size < 1024:
            return f'{int(binary_size)} bytes'
        elif binary_size < 1024 * 1024:
            return f'{binary_size/1024:.1f} KB'
        else:
            return f'{binary_size/(1024*1024):.1f} MB'
    
    def _format_notebook(self, cells: List[NotebookCell], metadata: Dict) -> str:
        """Format processed cells into readable output."""
        lines = []
        
        # Header with metadata
        lines.append('=' * 60)
        lines.append('JUPYTER NOTEBOOK')
        lines.append('=' * 60)
        
        kernel = metadata.get('kernelspec', {}).get('display_name', 'Unknown')
        language = metadata.get('kernelspec', {}).get('language', 'python')
        lines.append(f'Kernel: {kernel}')
        lines.append(f'Language: {language}')
        lines.append(f'Total cells: {len(cells)}')
        lines.append('=' * 60)
        lines.append('')
        
        # Process each cell
        for i, cell in enumerate(cells, 1):
            lines.append(f'─' * 60)
            
            if cell.cell_type == 'code':
                lines.append(f'[Cell {i}] Code' + 
                           (f' [In]: {cell.execution_count}' if cell.execution_count else ''))
                lines.append('─' * 60)
                lines.append(cell.source)
                
                if cell.output_summary:
                    lines.append('')
                    lines.append('[Output]:')
                    lines.append(cell.output_summary)
            
            elif cell.cell_type == 'markdown':
                lines.append(f'[Cell {i}] Markdown')
                lines.append('─' * 60)
                lines.append(cell.source)
            
            elif cell.cell_type == 'raw':
                lines.append(f'[Cell {i}] Raw')
                lines.append('─' * 60)
                lines.append(cell.source)
            
            lines.append('')
        
        lines.append('=' * 60)
        lines.append('END OF NOTEBOOK')
        lines.append('=' * 60)
        
        return '\n'.join(lines)


# Integration with OHEditor
class OHEditorNotebookMixin:
    """Mixin to add notebook support to OHEditor."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._notebook_handler = NotebookHandler()
    
    def view(self, path: Union[str, Path], view_range: Optional[list] = None) -> str:
        """Extended view method with notebook support."""
        path = Path(path)
        
        # Special handling for notebooks
        if self._notebook_handler.is_notebook(path):
            return self._view_notebook(path)
        
        # Fall back to standard view
        return super().view(path, view_range)
    
    def _view_notebook(self, path: Path) -> str:
        """View a notebook with proper formatting."""
        try:
            content = self._notebook_handler.process_notebook(path)
            # Return as a result object similar to OHEditor
            return self._create_view_result(path, content)
        except json.JSONDecodeError as e:
            raise FileValidationError(
                path=str(path),
                message=f"Invalid Jupyter notebook JSON: {e}"
            )
        except Exception as e:
            raise FileValidationError(
                path=str(path),
                message=f"Error processing notebook: {e}"
            )
    
    def _create_view_result(self, path: Path, content: str):
        """Create a view result object (adapt to your OHEditor result type)."""
        # This should match your OHEditor's result structure
        class ViewResult:
            def __init__(self, output, path):
                self.output = output
                self.path = str(path)
        
        return ViewResult(content, path)


def patch_oh_editor_for_notebooks(editor_class):
    """
    Patch OHEditor class to support notebooks.
    
    Usage:
        from openhands_aci.editor.editor import OHEditor
        patch_oh_editor_for_notebooks(OHEditor)
    """
    original_view = editor_class.view
    notebook_handler = NotebookHandler()
    
    def patched_view(self, path, view_range=None):
        path = Path(path)
        
        if notebook_handler.is_notebook(path):
            try:
                content = notebook_handler.process_notebook(path)
                # Return in expected format
                if hasattr(self, '_create_view_result'):
                    return self._create_view_result(path, content)
                else:
                    # Simple dict return
                    return {'output': content, 'path': str(path)}
            except Exception as e:
                from openhands_aci.editor.errors import FileValidationError
                raise FileValidationError(
                    path=str(path),
                    message=f"Notebook processing error: {e}"
                )
        
        return original_view(self, path, view_range)
    
    editor_class.view = patched_view
    return editor_class
