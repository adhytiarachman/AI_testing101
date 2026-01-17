"""
Batch-crawls URLs in parallel and SAVES OUTPUT TO FILES (Updated - Clean Version).
Creates timestamped folder with: crawl_results.json + individual .md files.
"""
import os
import sys
import psutil
import asyncio
import json
from typing import List
from datetime import datetime
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
import pathlib

async def crawl_parallel(urls: List[str], max_concurrent: int = 10):
    print("\n=== Parallel Crawling with FILE OUTPUT ===")
    
    # Auto-create timestamped output directory
    output_dir = f"crawl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pathlib.Path(output_dir).mkdir(exist_ok=True)
    print(f" Saving results to: {output_dir}/")
    
    # Track peak memory usage
    peak_memory = 0
    process = psutil.Process(os.getpid())
    def log_memory(prefix: str = ""):
        nonlocal peak_memory
        current_mem = process.memory_info().rss
        if current_mem > peak_memory:
            peak_memory = current_mem
        print(f"{prefix} Memory: {current_mem // (1024*1024)} MB, Peak: {peak_memory // (1024*1024)} MB")

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    all_results = []
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        log_memory("Before crawl: ")
        
        results = await crawler.arun_many(
            urls=urls,
            config=crawl_config,
            dispatcher=dispatcher
        )
        
        success_count = 0
        fail_count = 0
        
        for result in results:
            # Safe handling of failed crawls
            result_data = {
                "url": result.url,
                "success": result.success,
                "timestamp": datetime.now().isoformat(),
            }
            
            if result.success and hasattr(result.markdown, 'raw_markdown'):
                success_count += 1
                markdown_content = result.markdown.raw_markdown or ""
                
                result_data.update({
                    "markdown_length": len(markdown_content),
                    "markdown_preview": markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content,
                    "title": getattr(result.markdown, 'title', 'No title')
                })
                
                # Safe filename from URL
                safe_filename = result.url.split('/')[-1].replace('?', '_').replace('&', '_')[:50] or "homepage"
                if safe_filename == '/':
                    safe_filename = "homepage"
                safe_filename += ".md"
                
                # Save Markdown file
                md_path = os.path.join(output_dir, safe_filename)
                try:
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    print(f" SAVED: {md_path}")
                except Exception as e:
                    print(f"⚠️  Failed to save MD: {e}")
                    
            else:
                fail_count += 1
                result_data["markdown_length"] = 0
                result_data["error"] = getattr(result, 'error_message', 'Unknown error')
                print(f"❌ Failed: {result.url} - {result_data['error']}")
            
            all_results.append(result_data)

        # Safe JSON summary
        summary = {
            "crawl_timestamp": datetime.now().isoformat(),
            "total_urls": len(urls),
            "success_count": success_count,
            "fail_count": fail_count,
            "peak_memory_mb": peak_memory // (1024 * 1024),
            "output_directory": output_dir,
            "results": all_results
        }
        
        json_path = os.path.join(output_dir, "crawl_results.json")
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"\n SAVED SUMMARY: {json_path}")
        except Exception as e:
            print(f"❌ JSON save failed: {e}")
        
        log_memory("After crawl: ")
        print(f"\n SUMMARY: {success_count} successful, {fail_count} failed")
        print(f" Files in: {output_dir}/")

async def main():
    urls = ["https://www.docling.ai/", "https://numpy.org/doc/stable/user/quickstart.html", "https://www.tensorflow.org/api_docs/python/tf"]
    print(f" Crawling {len(urls)} URLs...")
    await crawl_parallel(urls, max_concurrent=5)

if __name__ == "__main__":
    asyncio.run(main())
