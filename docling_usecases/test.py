from docling.document_converter import DocumentConverter

source = "F:/AI_testing101/docling_usecases/documents/2408.09869v5.pdf"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"