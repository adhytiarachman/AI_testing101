## D
![](https://www.docling.ai/img/logo.svg)
## CLING
  * [Start](https://www.docling.ai/#start)
  * [Features](https://www.docling.ai/#features)
  * [ Docs ](https://docling-project.github.io/docling "Documentation")[ Chat ](https://app.dosu.dev/097760a8-135e-4789-8234-90c8837d7f1c/ask?utm_source=github "Chat with Dosu")

[ ![](https://www.docling.ai/img/github.svg) ](https://github.com/docling-project/docling "GitHub")
###  Docling converts messy documents into structured data and simplifies downstream document and AI processing by detecting tables, formulas, reading order, OCR, and much more. 
![](https://www.docling.ai/img/bigduck.webp)
Find us at 
[GitHub](https://docling-project.github.io/docling) [HuggingFace](https://huggingface.co/docling-project)
[Discord](https://www.docling.ai/discord) [LinkedIn](https://linkedin.com/company/docling) [YouTube](https://www.youtube.com/playlist?list=PLt0drfpBaTa1ywCtPwJGLYg-t0UmxhQP4)
### Start
**Install** Docling as a [Python library](https://pypi.org/project/docling) with your favorite package manager: 
```
pip install docling
```

**Use** the CLI directly from your terminal: 
```
docling https://arxiv.org/pdf/2206.01062
```

**Integrate** a document conversion into your Python application: 
```
from docling.document_converter import DocumentConverter

source = "https://arxiv.org/pdf/2408.09869"
converter = DocumentConverter()
doc = converter.convert(source).document
print(doc.export_to_markdown())

```

**Explore** the [examples](https://docling-project.github.io/docling/examples)
### Features
**Parse** many document formats into a unified and structured form. 
PDF Word PowerPoint Excel
Markdown HTML AsciiDoc CSV WebVTT MP3 WAV
PNG JPEG TIFF BMP WEBP
**Access** document components and their properties in the [Docling Document](https://docling-project.github.io/docling/concepts/docling_document). 
**Export** a parsed document to formats that simplify processing and ingestion into AI, RAG, and agentic systems. 
Text Markdown HTML
JSON [Doctags](https://arxiv.org/pdf/2503.11576)
![](https://www.docling.ai/img/lf.svg) Copyright © Docling, a Series of LF Projects, LLC.   
[terms of use](https://lfprojects.org/policies/terms-of-use) | [trademark policy](https://lfprojects.org/policies/trademark-policy) | [general policies](https://lfprojects.org/policies/general-rules-of-operation-policy)
