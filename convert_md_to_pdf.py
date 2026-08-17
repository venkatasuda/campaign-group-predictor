#!/usr/bin/env python
"""Convert markdown file to PDF"""

import markdown
from fpdf import FPDF
from pathlib import Path
import sys
import re
from html.parser import HTMLParser

class MarkdownToText(HTMLParser):
    """Simple HTML to text converter for markdown content"""
    def __init__(self):
        super().__init__()
        self.text = []
        self.in_code = False
        self.in_pre = False
        self.current_heading_level = 0
        
    def handle_data(self, data):
        if data.strip():
            self.text.append(data.strip())
    
    def handle_starttag(self, tag, attrs):
        if tag == 'h1':
            self.text.append('\n')
            self.current_heading_level = 1
        elif tag == 'h2':
            self.text.append('\n')
            self.current_heading_level = 2
        elif tag == 'h3':
            self.text.append('\n')
            self.current_heading_level = 3
        elif tag == 'h4':
            self.text.append('\n')
            self.current_heading_level = 4
        elif tag in ['p', 'li']:
            self.text.append('\n')
        elif tag == 'br':
            self.text.append('\n')
        elif tag == 'code':
            self.in_code = True
            self.text.append('[CODE]')
        elif tag == 'pre':
            self.in_pre = True
            self.text.append('\n')
        
    def handle_endtag(self, tag):
        if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            self.text.append('\n')
            self.current_heading_level = 0
        elif tag == 'code':
            self.in_code = False
            self.text.append('[/CODE]')
        elif tag == 'pre':
            self.in_pre = False
            self.text.append('\n')
        elif tag in ['p', 'ul', 'ol', 'blockquote']:
            self.text.append('\n')

def convert_markdown_to_pdf(md_file_path, output_pdf_path=None):
    """
    Convert a markdown file to PDF.
    
    Args:
        md_file_path: Path to the markdown file
        output_pdf_path: Path for the output PDF (defaults to same name with .pdf extension)
    """
    md_file = Path(md_file_path)
    
    if not md_file.exists():
        print(f"Error: File {md_file_path} not found")
        return False
    
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['extra', 'toc'])
    
    # Parse HTML to text
    parser = MarkdownToText()
    parser.feed(html_content)
    text_content = ' '.join(parser.text)
    
    # Clean up the text
    text_content = re.sub(r'\s+', ' ', text_content)
    text_content = re.sub(r'\n\s+\n', '\n\n', text_content)
    
    # Determine output path
    if output_pdf_path is None:
        output_pdf_path = md_file.with_suffix('.pdf')
    
    output_pdf = Path(output_pdf_path)
    
    # Create PDF
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Use a font that supports Unicode
        # Try to use Helvetica which is a default font
        try:
            pdf.set_font("helvetica", size=11)
        except:
            # Fallback to courier if helvetica is not available
            pdf.set_font("courier", size=11)
        
        # Add title
        try:
            pdf.set_font("helvetica", "B", size=14)
        except:
            pdf.set_font("courier", "B", size=14)
        
        title = md_file.stem
        # Replace problematic Unicode characters
        title = title.replace("—", "-").replace("–", "-").replace("…", "...")
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        
        try:
            pdf.set_font("helvetica", size=10)
        except:
            pdf.set_font("courier", size=10)
        
        pdf.ln(5)
        
        # Add content - truncate to reasonable length
        lines = text_content.split('\n')[:100]  # Limit to first 100 lines
        for line in lines[:500]:  # Further limit lines for PDF size
            if line.strip():
                # Replace problematic Unicode characters
                line = line.replace("—", "-").replace("–", "-").replace("…", "...")
                line = line.replace("'", "'").replace("'", "'")
                line = line.replace(""", '"').replace(""", '"')
                
                # Keep only ASCII and basic extended ASCII
                line = ''.join(c if ord(c) < 256 else '?' for c in line)
                
                # Wrap long lines
                try:
                    pdf.multi_cell(0, 5, line[:200], align='L')  # Limit line length
                except Exception as e:
                    print(f"Warning: Could not add line: {e}")
        
        pdf.output(str(output_pdf))
        print(f"✓ Successfully converted '{md_file}' to '{output_pdf}'")
        return True
    except Exception as e:
        print(f"Error converting to PDF: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_md_to_pdf.py <markdown_file> [output_pdf_file]")
        sys.exit(1)
    
    md_file = sys.argv[1]
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_markdown_to_pdf(md_file, pdf_file)
    sys.exit(0 if success else 1)
