from typing import Dict, Any, Optional, List
import os
import re
from datetime import datetime
import PyPDF2
import pdfplumber
from .utils import TextProcessor

class PaperParser:
    def __init__(self):
        self.text_processor = TextProcessor()
        
    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        
        metadata = self._extract_metadata(pdf_path)
        
        
        text_content = self._extract_text(pdf_path)
        
        
        sections = self._extract_sections(text_content)
        
        
        references = self._extract_references(text_content)
        
        
        figures_tables = self._extract_figures_tables(text_content)
        
        return {
            "title": metadata.get("title", ""),
            "authors": metadata.get("authors", []),
            "publication_date": metadata.get("publication_date"),
            "journal": metadata.get("journal", ""),
            "abstract": sections.get("abstract", ""),
            "introduction": sections.get("introduction", ""),
            "methods": sections.get("methods", ""),
            "results": sections.get("results", ""),
            "discussion": sections.get("discussion", ""),
            "conclusion": sections.get("conclusion", ""),
            "references": references,
            "figures_tables": figures_tables,
            "metadata": metadata
        }
    
    def _extract_metadata(self, pdf_path: str) -> Dict[str, Any]:

        metadata = {}
        
        try:
            with open(pdf_path, 'rb') as file:
                
                pdf_reader = PyPDF2.PdfReader(file)
                info = pdf_reader.metadata
                
                if info:
                    metadata.update({
                        "title": info.get("/Title", ""),
                        "authors": info.get("/Author", "").split(";") if info.get("/Author") else [],
                        "creation_date": info.get("/CreationDate", ""),
                        "modification_date": info.get("/ModDate", ""),
                        "producer": info.get("/Producer", ""),
                        "creator": info.get("/Creator", "")
                    })
                
                
                with pdfplumber.open(pdf_path) as pdf:
                    first_page = pdf.pages[0]
                    text = first_page.extract_text()
                    
                    
                    journal_match = re.search(r"Published in:?\s*([^\n]+)", text)
                    if journal_match:
                        metadata["journal"] = journal_match.group(1).strip()
                    
                    
                    date_match = re.search(r"Published:?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})", text)
                    if date_match:
                        try:
                            metadata["publication_date"] = datetime.strptime(
                                date_match.group(1),
                                "%Y-%m-%d"
                            )
                        except ValueError:
                            pass
                    
                    
                    doi_match = re.search(r"DOI:?\s*(10\.\d{4,}/[\w\.-]+)", text)
                    if doi_match:
                        metadata["doi"] = doi_match.group(1).strip()
                    
                    
                    keywords_match = re.search(r"Keywords:?\s*([^\n]+)", text)
                    if keywords_match:
                        metadata["keywords"] = [
                            k.strip() for k in keywords_match.group(1).split(",")
                        ]
        
        except Exception as e:
            print(f"Error extracting metadata from {pdf_path}: {str(e)}")
        
        return metadata
    
    def _extract_text(self, pdf_path: str) -> str:

        text = ""
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
        
        return text
    
    def _extract_sections(self, text: str) -> Dict[str, str]:

        sections = {}
        
        
        section_patterns = {
            "abstract": r"(?i)abstract\s*",
            "introduction": r"(?i)introduction\s*",
            "methods": r"(?i)(?:methods|materials and methods)\s*",
            "results": r"(?i)results\s*",
            "discussion": r"(?i)discussion\s*",
            "conclusion": r"(?i)conclusion\s*"
        }
        
        
        current_section = None
        current_content = []
        
        for line in text.split("\n"):
            
            for section_name, pattern in section_patterns.items():
                if re.match(pattern, line.strip()):
                    if current_section:
                        sections[current_section] = "\n".join(current_content)
                    current_section = section_name
                    current_content = []
                    break
            else:
                if current_section:
                    current_content.append(line)
        
        
        if current_section and current_content:
            sections[current_section] = "\n".join(current_content)
        
        return sections
    
    def _extract_references(self, text: str) -> List[Dict[str, str]]:

        references = []
        
        
        ref_section_match = re.search(r"(?i)references\s*(.*?)(?=\n\s*\n|\Z)", text, re.DOTALL)
        if ref_section_match:
            ref_text = ref_section_match.group(1)
            
            
            ref_items = re.split(r"\n\s*\d+\.\s*", ref_text)
            
            for ref in ref_items:
                if ref.strip():
                    
                    ref_info = {
                        "text": ref.strip(),
                        "authors": self._extract_reference_authors(ref),
                        "year": self._extract_reference_year(ref),
                        "title": self._extract_reference_title(ref),
                        "journal": self._extract_reference_journal(ref),
                        "doi": self._extract_reference_doi(ref)
                    }
                    references.append(ref_info)
        
        return references
    
    def _extract_figures_tables(self, text: str) -> Dict[str, List[Dict[str, Any]]]:

        figures_tables = {
            "figures": [],
            "tables": []
        }
        
        
        figure_pattern = r"(?i)Figure\s+(\d+)[.:]\s*([^\n]+)"
        table_pattern = r"(?i)Table\s+(\d+)[.:]\s*([^\n]+)"
        
        
        for match in re.finditer(figure_pattern, text):
            figure_info = {
                "number": match.group(1),
                "title": match.group(2).strip(),
                "type": "figure"
            }
            figures_tables["figures"].append(figure_info)
        
        
        for match in re.finditer(table_pattern, text):
            table_info = {
                "number": match.group(1),
                "title": match.group(2).strip(),
                "type": "table"
            }
            figures_tables["tables"].append(table_info)
        
        return figures_tables
    
    def _extract_reference_authors(self, ref_text: str) -> List[str]:
        authors = []
        author_match = re.search(r"^([^\(]+?)\s*\(", ref_text)
        if author_match:
            authors = [a.strip() for a in author_match.group(1).split(",")]
        return authors
    
    def _extract_reference_year(self, ref_text: str) -> Optional[int]:
        year_match = re.search(r"\((\d{4})\)", ref_text)
        if year_match:
            try:
                return int(year_match.group(1))
            except ValueError:
                pass
        return None
    
    def _extract_reference_title(self, ref_text: str) -> str:
        title_match = re.search(r"\"([^\"]+)\"", ref_text)
        if title_match:
            return title_match.group(1)
        return ""
    
    def _extract_reference_journal(self, ref_text: str) -> str:

        journal_match = re.search(r"([^\.]+)\.\s*\d{4}", ref_text)
        if journal_match:
            return journal_match.group(1).strip()
        return ""
    
    def _extract_reference_doi(self, ref_text: str) -> str:
        doi_match = re.search(r"10\.\d{4,}/[\w\.-]+", ref_text)
        if doi_match:
            return doi_match.group(0)
        return "" 