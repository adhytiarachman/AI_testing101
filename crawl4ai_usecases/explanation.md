# Brief Explaination of Codes:

The folder contains five Python scripts. While they address a similar problem at a high level, they are not the same; each script follows a different working principle and implements a distinct approach. The five scripts are:

### test.py:

It is the same code which is mentioned in the crawl4ai website. 

### ex1.py

This script fetches URLs from the website and crawls them sequentially while reusing a single browser session.

##### Key Components

**BrowserConfig** sets up the browser environment with `headless=True` for background operation and includes performance optimization flags like `--disable-gpu`, `--disable-dev-shm-usage`, and `--no-sandbox`. These flags are particularly useful for Docker or low-memory environments.

**Session Reuse** through `session_id="session1"` maintains the same browser context across all URLs, avoiding the overhead of repeatedly opening and closing browser tabs. This approach is faster for sequential workflows because the browser state persists between requests.

**DefaultMarkdownGenerator** converts the crawled HTML content into Markdown format, making it easier to process for LLMs or text analysis.

The script uses `crawler.start()` and `crawler.close()` to manually manage the browser lifecycle, giving explicit control over when resources are allocated and released.

### ex2.py:

This script crawls multiple URLs simultaneously using `arun_many()` with intelligent resource management and memory monitoring.

##### Advanced Features

**MemoryAdaptiveDispatcher** automatically controls concurrency based on system memory usage, pausing new requests if memory exceeds 70% and limiting parallel browser sessions to the `max_concurrent` value. This prevents system overload during large-scale crawling operations.

**Memory Tracking** uses the `psutil` library to monitor current and peak memory consumption throughout the crawl, providing observability for resource-intensive operations.

**Batch Processing** with `arun_many()` processes all URLs in parallel rather than sequentially, significantly reducing total crawl time for large URL lists. The `CacheMode.BYPASS` setting ensures fresh data is fetched on every request.

The script prints a summary of successful and failed crawls, making it easy to identify problematic URLs.

### ex3.py:

This script scrapes a Markdown file and splits it into logical sections based on heading levels (# and ##).

##### Content Processing

The script uses regular expressions to identify all `#` and `##` headers in the Markdown content, then splits the document at these boundaries to create discrete chunks. This chunking approach is useful for processing long documentation files in smaller, topic-focused segments that can be fed to LLMs or stored separately.

Each chunk is printed with its index, allowing inspection of how the document was segmented. The target URL points to a `.txt` file containing Markdown-formatted content from website.

### ex4.py:

This script performs depth-first crawling by discovering and following internal links up to a specified depth level.

##### Recursive Crawling Logic

**URL Normalization** removes fragments (the part after `#`) using `urldefrag()` to prevent treating `page.html` and `page.html#section` as different pages.

**Depth-Based Exploration** crawls all discovered URLs at the current depth in parallel before moving to the next level, maintaining a breadth-first traversal pattern for `max_depth` levels.

**Deduplication** tracks visited URLs in a set to prevent infinite loops and redundant crawling, which is critical for sites with complex internal linking structures.[]

**Link Extraction** accesses `result.links.get("internal", [])` to retrieve only same-domain links from each crawled page, ensuring the crawler stays within the target site.

The combination of `MemoryAdaptiveDispatcher` and batch processing with `arun_many()` allows efficient parallel crawling at each depth level while respecting system resource constraints.

### Q: When what to use?

| Scenario                            | Use Example         | Key Advantage                 | Best For                                  |
| ----------------------------------- | ------------------- | ----------------------------- | ----------------------------------------- |
| Small-medium URL lists (sequential) | **Example 1** | Session reuse, low overhead   | Limited resources, Docker, simple batches |
| Large URL lists (parallel)          | **Example 2** | Memory-adaptive, auto-scaling | High-volume scraping, production jobs     |
| Single document processing          | **Example 3** | Header-based chunking         | Markdown splitting, LLM prep              |
| Site-wide exploration               | **Example 4** | Recursive internal links      | Full site crawling, discovery             |


##### Example 1: Sequential with Session Reuse

**Use when:**

* Crawling 10-100 URLs sequentially
* Running in Docker/low-memory environments
* Need predictable resource usage

**Why it works:**

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-normal bg-subtler"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden rounded-full border-subtlest ring-subtlest divide-subtlest bg-base"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtler"></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-subtle py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">text</div></div><div><span><code><span><span>session_id = "session1"  # Single browser context reused
</span></span><span>await crawler.start() → crawl all → await crawler.close()
</span><span></span></code></span></div></div></div></pre>

* One browser instance for all URLs[](https://docs.crawl4ai.com/advanced/session-management/)
* `--disable-gpu --no-sandbox` flags optimize for containers[](https://docs.crawl4ai.com/)
* Perfect for stable, controlled workflows

##### Example 2: Parallel with Memory Management

**Use when:**

* Crawling 100+ URLs simultaneously
* Need production-grade resource control
* Want success/failure summaries

**Key innovation:**

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-normal bg-subtler"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden rounded-full border-subtlest ring-subtlest divide-subtlest bg-base"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtler"></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-subtle py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">text</div></div><div><span><code><span><span>MemoryAdaptiveDispatcher(memory_threshold_percent=70.0)
</span></span><span>arun_many(urls=urls, dispatcher=dispatcher)
</span><span></span></code></span></div></div></div></pre>

* Auto-pauses at 70% memory usage
* `psutil` tracks peak memory for observability
* Scales `max_concurrent` dynamically

##### Example 3: Markdown Chunking

**Use when:**

* Processing single long Markdown documents
* Preparing content for RAG/LLM chunking
* Need header-based segmentation

**How it splits:**

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-normal bg-subtler"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden rounded-full border-subtlest ring-subtlest divide-subtlest bg-base"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtler"></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-subtle py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">python</div></div><div><span><code><span><span>header_pattern </span><span class="token token operator">=</span><span> re</span><span class="token token punctuation">.</span><span class="token token">compile</span><span class="token token punctuation">(</span><span class="token token">r'^(# .+|## .+)$'</span><span class="token token punctuation">,</span><span> re</span><span class="token token punctuation">.</span><span>MULTILINE</span><span class="token token punctuation">)</span><span>
</span></span><span><span></span><span class="token token"># Creates chunks: "# Header" → next "# Header"</span><span>
</span></span><span></span></code></span></div></div></div></pre>

* Perfect for documentation splitting
* Each chunk = one topic/section
* No parallel overhead needed

Example 4: Recursive Site Crawling

**Use when:**

* Need to discover/explore entire websites
* Start from homepage, follow internal links
* Multi-level depth crawling (up to `max_depth=3`)

**Smart features:**

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-normal bg-subtler"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden rounded-full border-subtlest ring-subtlest divide-subtlest bg-base"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtler"></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-subtle py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">text</div></div><div><span><code><span><span>visited = set()  # Deduplication
</span></span><span>normalize_url()  # Removes #fragments
</span><span>result.links.get("internal", [])  # Same-domain only
</span><span></span></code></span></div></div></div></pre>

* Breadth-first by depth level
* Prevents infinite loops
* Parallel at each depth

**Resource Usage Guide:**

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-normal bg-subtler"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden rounded-full border-subtlest ring-subtlest divide-subtlest bg-base"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtler"></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-subtle py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">text</div></div><div><span><code><span><span>Low Resources → Example 1 (sequential)
</span></span><span>High Throughput → Example 2 (parallel + memory mgmt)
</span><span>Single File → Example 3 (chunking)
</span><span>Site Discovery → Example 4 (recursive)</span></code></span></div></div></div></pre>
