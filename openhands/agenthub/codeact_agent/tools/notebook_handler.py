import json
from typing import List, Dict, Optional, Tuple

class NotebookHandler:
    def __init__(self, max_source_lines: int = 50, max_output_chars: int = 1000):
        self.max_source_lines = max_source_lines
        self.max_output_chars = max_output_chars

    def parse_notebook(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cells = data.get('cells', [])
            result = []
            for i, cell in enumerate(cells):
                cell_type = cell.get('cell_type', 'unknown')
                result.append(f"--- Cell {i} ({cell_type}) ---")
                
                # Обработка кода
                source = "".join(cell.get('source', []))
                if len(source.splitlines()) > self.max_source_lines:
                    source = "\n".join(source.splitlines()[:self.max_source_lines]) + "\n... [source truncated]"
                result.append(source)
                
                # Обработка вывода (удаление base64 и обрезка)
                outputs = cell.get('outputs', [])
                processed_output, truncated = self._process_outputs(outputs)
                if processed_output:
                    result.append(f"Output:\n{processed_output}")
                    if truncated:
                        result.append("... [output truncated]")
            
            return "\n".join(result)
        except Exception as e:
            return f"Error parsing notebook: {str(e)}"

    def _process_outputs(self, outputs: List[Dict]) -> Tuple[Optional[str], bool]:
        full_output = []
        is_truncated = False
        
        for out in outputs:
            # Игнорируем картинки (base64)
            if 'data' in out:
                if any(k.startswith('image/') for k in out['data'].keys()):
                    continue
                
                text_data = out['data'].get('text/plain', [])
                full_output.append("".join(text_data))
            
            elif 'text' in out:
                full_output.append("".join(out['text']))
                
        text_result = "\n".join(full_output)
        if len(text_result) > self.max_output_chars:
            text_result = text_result[:self.max_output_chars]
            is_truncated = True
            
        return text_result if text_result else None, is_truncated
