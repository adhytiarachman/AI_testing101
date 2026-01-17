# Building RAG Agents with LLMs

Introduction

# Building RAG Agents with LLMs

Introduction

<!-- image -->

# Large Language Models

Backbones for Language Understanding

Backbones for language tasks, including classification and generation.

https://www.nvidia.com/content/dam/en-zz/Solutions/lp/large-language-models-ebook/nvidia-llm-ebook-og.jpg

# Dialog/Retrieval Agents

LLMs with Context and Control

https://www.nvidia.com/content/dam/en-zz/Solutions/lp/large-language-models-ebook/nvidia-llm-ebook-og.jpg

<!-- image -->

<!-- image -->

User Asks Something

Agent Responds

# Dialog/Retrieval Agents

LLMs with Context and Control

https://www.nvidia.com/content/dam/en-zz/Solutions/lp/large-language-models-ebook/nvidia-llm-ebook-og.jpg

<!-- image -->

<!-- image -->

LLM Orchestration: Software + LLM helps to route to software and LLMs.

User Asks Something

Agent Responds

# Dialog/Retrieval Agents

LLMs with Context and Control

https://www.nvidia.com/content/dam/en-zz/Solutions/lp/large-language-models-ebook/nvidia-llm-ebook-og.jpg

<!-- image -->

<!-- image -->

LLM Orchestration: Software + LLM helps to route to software and LLMs.

Retrieval: Tool runs algorithms (database, code execution, semantic search, return a constant value, etc) to provide context

User Asks Something

Agent Responds

# Dialog/Retrieval Agents

LLMs with Context and Control

https://www.nvidia.com/content/dam/en-zz/Solutions/lp/large-language-models-ebook/nvidia-llm-ebook-og.jpg

<!-- image -->

<!-- image -->

LLM Orchestration: Software + LLM helps to route to software and LLMs.

Retrieval: Tool runs algorithms (database, code execution, semantic search, return a constant value, etc) to provide context

User Asks Something

Agent Responds

Augmented: Based on tool responses, the software pipeline synthesizes some “context”

to feed to LLM w/ question.

# Dialog/Retrieval Agents

LLMs with Context and Control

https://www.nvidia.com/content/dam/en-zz/Solutions/lp/large-language-models-ebook/nvidia-llm-ebook-og.jpg

<!-- image -->

<!-- image -->

LLM Orchestration: Software + LLM helps to route to software and LLMs.

Retrieval: Tool runs algorithms (database, code execution, semantic search, return a constant value, etc) to provide context

Generation: Based on question, instructions, and enhanced context, the LLM returns a response.

User Asks Something

Agent Responds

Augmented: Based on tool responses, the software pipeline synthesizes some “context”

to feed to LLM w/ question.

# Chat Applications

Full Applications Build with LLMs

https://www.nvidia.com/content/dam/en-zz/Solutions/lp/large-language-models-ebook/nvidia-llm-ebook-og.jpg

<!-- image -->

# Chat Applications

Full Applications Build with LLMs

Production-Ready APIs That Run Anywhere | NVIDIA

<!-- image -->

# Chat Applications

Full Applications Build with LLMs

Production-Ready APIs That Run Anywhere | NVIDIA

<!-- image -->

# Prerequisites

- Prior LLM/LangChain Exposure
- Intermediate Python Experience
- Exposure to Web Engineering

RAG Agents in Production

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

# Course Objectives

- Environment
- LLM Services
- Intro to LangChain (LCEL)
- Running State Chains
- Document Loading
- Embeddings
- Document Retrieval
- RAG Evaluation

RAG Agents in Production

<!-- image -->

# Course Objectives

- (Starts at 9)
- Environment
- LLM Services
- Intro to LangChain (LCEL)
- Running State Chains
- (Lunch)
- Document Loading
- Embeddings
- Document Retrieval
- RAG Evaluation
- (Ends at 5, take-home/ask instructor)

RAG Agents in Production

<!-- image -->

# Building RAG Agents with LLMs

Part 1: Your Course Environment

# Typical Jupyter Labs Interface

<!-- image -->

jupyter lab . &amp;; open -a Safari http://localhost:8888/lab 

# Typical Jupyter Labs Interface

<!-- image -->

jupyter lab . &amp;; open -a Safari http://localhost:8888/lab 

Your Device

         Your Device

Web Browser

Files

OS

Hardware

C++

Python

# Typical Jupyter Labs Interface

<!-- image -->

jupyter lab . &amp;; open -a Safari http://localhost:8888/lab 

Your Device

         Your Device

:8888

Web Browser

Jupyter Labs

Files

OS

Hardware

C++

Python

# Typical Jupyter Labs Interface

<!-- image -->

<!-- image -->

https://colab.research.google.com

jupyter lab . &amp;; open -a Safari http://localhost:8888/lab 

Your Device

         Your Device

:8888

Web Browser

Jupyter Labs

Files

OS

Hardware

C++

Python

# Typical Jupyter Labs Interface

<!-- image -->

<!-- image -->

https://colab.research.google.com

jupyter lab . &amp;; open -a Safari http://localhost:8888/lab 

Your Device

         Your Device

:8888

Web Browser

Jupyter Labs

Files

OS

Hardware

C++

Python

Your Device

Web Browser

Files

OS

Hardware

C++

Python

<!-- image -->

# Typical Jupyter Labs Interface

<!-- image -->

<!-- image -->

https://colab.research.google.com

jupyter lab . &amp;; open -a Safari http://localhost:8888/lab 

Your Device

         Your Device

:8888

Web Browser

Jupyter Labs

Files

OS

Hardware

C++

Python

Your Device

Web Browser

Files

OS

Hardware

C++

Python

<!-- image -->

<!-- image -->

<!-- image -->

# DLI Jupyter Labs Interface

<!-- image -->

# DLI Jupyter Labs Interface

<!-- image -->

<!-- image -->

<!-- image -->

Your Device

Your Device

Web Browser

Your Device

Remote Host

Jupyter Labs

<!-- image -->

<!-- image -->

Files

OS

Hardware

C++

Python

# DLI Jupyter Labs Interface

<!-- image -->

<!-- image -->

<!-- image -->

Your Device

Your Device

Web Browser

Your Device

Remote Host

Jupyter Labs

<!-- image -->

<!-- image -->

Files

OS

Hardware

C++

Python

Good to go!

… sort of …

# DLI Jupyter Labs Interface

<!-- image -->

<!-- image -->

Your Device

Your Device

Web Browser

Your Device

Remote Host

Jupyter Labs

<!-- image -->

<!-- image -->

Files

OS

Hardware

C++

Python

Data Loader

Shell

Proxy Service

Node

Scheduler

Env

Python

More Processes?

# DLI Jupyter Labs Interface

<!-- image -->

<!-- image -->

Your Device

Your Device

Web Browser

Your Device

Remote Host

Jupyter Labs

<!-- image -->

<!-- image -->

Files

OS

Hardware

C++

Python

Data Loader

Shell

Proxy Service

Node

Scheduler

Env

Python

More Processes?

Dividing Resources?

# Containerization with Docker

Your Device

Host

Jupyter Labs

Files

OS

Hardware

C++

Python

Data Loader

Shell

Proxy Service

Node

Scheduler

Env

Python

Your Device

Host

Files

OS

Hardware

C++

Python

Jupyter

Python

Volume

Volume

GPU/2

Sh

:8888

:8070

:88

<!-- image -->

Compartmentalizing Functionality Into Microservices

# Microservices Workflow

Your Device

Your Device

Web Browser

# Microservices Workflow

Your Device

Your Device

Web Browser

Your Device

Remote Host

<!-- image -->

<!-- image -->

Files

OS

Hardware

C++

Python

1. Allocate Resources
2. Define Services
3. Construct Containers
4. Start Processes

# Microservices Workflow

Your Device

Your Device

Web Browser

Your Device

Remote Host

<!-- image -->

<!-- image -->

Files

OS

Hardware

C++

Python

1. Allocate Resources
2. Define Services
3. Construct Containers
4. Start Processes

Jupyter

Python

Volume

Volume

GPU/2

Sh

:8888

:8070

:88

# Microservices Workflow

Your Device

Your Device

Web Browser

Your Device

Remote Host

<!-- image -->

<!-- image -->

Files

OS

Hardware

C++

Python

1. Allocate Resources
2. Define Services
3. Construct Containers
4. Start Processes

Jupyter Notebook Environment

Database Environment

Data Loader

:8888

:8070

:88

# Microservices Workflow

Your Device

Your Device

Web Browser

Your Device

Remote Host

<!-- image -->

<!-- image -->

Files

OS

Hardware

C++

Python

1. Allocate Resources
2. Define Services
3. Construct Containers
4. Start Processes

Jupyter Notebook Environment

Database Environment

Data Loader

:8888

:8070

:88

# Scaling Containerized Applications

Your Device

Arbitrary Host

Files

OS

Hardware

C++

Python

Jupyter Notebook Environment

Database Environment

Data Loader

:8888

:8070

Company

Database

GenAI 

Service

<!-- image -->

<!-- image -->

Your Device

Your Device

Web Browser

<!-- image -->

Your Device

Other Devices

Web Browser

Your Device

Other Devices

Web Browser

Your Device

Other Devices

Web Browser

Your Device

Other Devices

Web Browser

# Scaling Containerized Applications

Your Device

Arbitrary Host

Files

OS

Hardware

C++

Python

Jupyter Notebook Environment

Database Environment

Data Loader

:8888

:8070

Company

Database

GenAI 

Service

<!-- image -->

<!-- image -->

Your Device

Your Device

Web Browser

Your Device

Other Devices

Web Browser

Your Device

Other Devices

Web Browser

Your Device

Other Devices

Web Browser

Your Device

Other Devices

<!-- image -->

<!-- image -->

<!-- image -->

Web Browser

Your Device

Your Device

Your Device

Your Device

Remote Host

Jupyter Notebook Server

Docker

Router

<!-- image -->

# Our Environment

Jupyter Notebooks

:88

‹#›

Your Device

Remote Host

Jupyter Notebook Server

Frontend

Docker

Router

<!-- image -->

<!-- image -->

# Our Environment

Jupyter Notebooks + Frontend

:8090

:88

‹#›

# Gradio

<!-- image -->

# Simple Gradio ChatInterface

<!-- image -->

<!-- image -->

https://huggingface.co/spaces/gradio/chatinterface\_streaming\_echo/blob/main/run.py

https://www.gradio.app/

# Gradio in HuggingFace Spaces

<!-- image -->

<!-- image -->

https://huggingface.co/spaces/camenduru-com/webui

# Custom Gradio Block Interface

<!-- image -->

https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks

<!-- image -->

Your Device

Remote Host

Jupyter Notebook Server

Frontend

Docker

Router

<!-- image -->

<!-- image -->

# Our Environment

Jupyter Notebooks + Frontend

:8090

:88

‹#›

# Building RAG Agents with LLMs

Part 2: LLM Services

Your Device

Remote Host

Jupyter Notebook Server

Frontend

Docker

Router

<!-- image -->

<!-- image -->

# Our Environment

Jupyter Notebooks + Frontend

:8090

:88

‹#›

# Dialog/Retrieval Agents

LLMs with Context and Control

Execute with dialog management and information retrieval in production

https://www.nvidia.com/content/dam/en-zz/Solutions/lp/large-language-models-ebook/nvidia-llm-ebook-og.jpg

<!-- image -->

<!-- image -->

# Dialog/Retrieval Agents

LLMs with Context and Control

Execute with dialog management and information retrieval in production

https://www.nvidia.com/content/dam/en-zz/Solutions/lp/large-language-models-ebook/nvidia-llm-ebook-og.jpg

<!-- image -->

<!-- image -->

# Standalone Environment LLM

Your Device

Remote Host

Jupyter Notebook Server

Frontend

Docker

Router

# Standalone Environment LLM

Your Device

Remote Host

Jupyter Notebook Server

Frontend

Docker

Router

LLM

# Standalone Environment LLM

<!-- image -->

   Jupyter Notebook 

H200

A100

   Jupyter Notebook 

A10

4070

   Jupyter Notebook 

VRAM-bound

CPU-only

&lt;s&gt;[INST]&lt;&lt;SYS&gt;&gt;

{{system\_message}}

&lt;&lt;/SYS&gt;&gt;

{{instruction}} [/INST] {{primer}}

&lt;s&gt;[INST]&lt;&lt;SYS&gt;&gt;

You are a code generator. 

Please provide Python code 

per the instruction.

&lt;&lt;/SYS&gt;&gt;

Write a Fibonacci method [/INST] ```python

## Implementation of Fibonacci w/

# Standalone Environment LLM

   Jupyter Notebook 

H200

A100

   Jupyter Notebook 

A10

4070

   Jupyter Notebook 

VRAM-bound

CPU-only

<!-- image -->

<!-- image -->

<!-- image -->

# Standalone Environment LLM

   Jupyter Notebook 

H200

A100

   Jupyter Notebook 

A10

4070

   Jupyter Notebook 

VRAM-bound

CPU-only

<!-- image -->

<!-- image -->

<!-- image -->

# Remote LLM Access

Your Device

Remote Host

Frontend

  Jupyter Notebook 

VRAM-bound

CPU-only

&lt;s&gt;[INST]&lt;&lt;SYS&gt;&gt;

{{system\_message}}

&lt;&lt;/SYS&gt;&gt;

{{instruction}} [/INST] {{primer}}

&lt;s&gt;[INST]&lt;&lt;SYS&gt;&gt;

You are a code generator. 

Please provide Python code 

per the instruction.

&lt;&lt;/SYS&gt;&gt;

Write a Fibonacci method [/INST] ```python

## Implementation of Fibonacci w/

<!-- image -->

<!-- image -->

Llama

Your Device

Remote Host

Frontend

  Jupyter Notebook 

VRAM-bound

CPU-only

# Large Model Hosting Platforms

NVIDIA 

GPU CLOUD

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

# Large Model Hosting Platforms

NVIDIA 

GPU CLOUD

<!-- image -->

<!-- image -->

1. GPT4/3.5
2. Integrated Tools/Multimodal Support
3. Models/Query Router Internal
4. Path to Local Deployment Less Clear

1. Publicly Available Models
2. Models/Query Router Accessible
3. Path to Local Deployment/Self-Hosting
4. Pieces to Compose Complex Systems

# Query Router Access

&lt;s&gt;[INST]&lt;&lt;SYS&gt;&gt;

{{system\_message}}

&lt;&lt;/SYS&gt;&gt;

{{instruction}} [/INST] {{primer}}????

<!-- image -->

Dalle-3

Embed

GPT4

<!-- image -->

<!-- image -->

<!-- image -->

Facilitate

Monitor

Optimize

Load Balance

# Query Router Access

&lt;s&gt;[INST]&lt;&lt;SYS&gt;&gt;

{{system\_message}}

&lt;&lt;/SYS&gt;&gt;

{{instruction}} [/INST] {{primer}}????

{

"messages": [{

  "content": "...",

  "role": "system"

},{

  "content": "...",

  "role": "user"

}],

"model": ”gpt-4”,

"temperature": 0.2,

"top\_p": 0.7,

"max\_tokens": 1024,

"stream": True

}

/chat/completions

<!-- image -->

<!-- image -->

Dalle-3

Embed

GPT4

<!-- image -->

<!-- image -->

<!-- image -->

https://kubernetes.io/

Facilitate

Monitor

Optimize

Load Balance

# Query Router Access

&lt;s&gt;[INST]&lt;&lt;SYS&gt;&gt;

{{system\_message}}

&lt;&lt;/SYS&gt;&gt;

{{instruction}} [/INST] {{primer}}????

{

"messages": [{

  "content": "...",

  "role": "system"

},{

  "content": "...",

  "role": "user"

}],

"model": ”gpt-4”,

"temperature": 0.2,

"top\_p": 0.7,

"max\_tokens": 1024,

"stream": True

}

/chat/completions

<!-- image -->

<!-- image -->

Dalle-3

Embed

GPT4

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

https://kubernetes.io/

OpenAI Gateway

<!-- image -->

embed-&lt;&gt;

<!-- image -->

Dalle-3

# Large Model Hosting Platforms - OpenAI

<!-- image -->

GPT

(3.5/4)

<!-- image -->

images/generate

chat/completion

completion

models

embeddings

  Responses

OpenAI Gateway

<!-- image -->

embed-&lt;&gt;

<!-- image -->

Dalle-3

Messages

Model Name

Settings

API Key

# Large Model Hosting Platforms - OpenAI

<!-- image -->

GPT

(3.5/4)

<!-- image -->

images/generate

chat/completion

completion

models

embeddings

  Responses

Messages

Model Name

Settings

API Key

# Large Model Hosting Platforms - OpenAI

https://developer.nvidia.com/blog/simplifying-ai-inference-in-production-with-triton/

OpenAI Gateway

<!-- image -->

GPT

(3.5/4)

<!-- image -->

embed-&lt;&gt;

images/generate

chat/completion

completion

models

embeddings

<!-- image -->

Dalle-3

<!-- image -->

End-User Application

Retriever Microservice

Server Management

# Large Model Hosting Platforms

NVIDIA 

GPU CLOUD

<!-- image -->

<!-- image -->

1. GPT4/3.5
2. Integrated Tools/Multimodal Support
3. Models/Query Router Internal
4. Path to Local Deployment Less Clear

1. Publicly Available Models
2. Models/Query Router Accessible
3. Path to Local Deployment/Self-Hosting
4. Pieces to Compose Complex Systems

# Large Model Hosting Platforms

NVIDIA 

GPU CLOUD

<!-- image -->

<!-- image -->

1. GPT4/3.5
2. Integrated Tools/Multimodal Support
3. Models/Query Router Internal
4. Path to Local Deployment Less Clear

1. Publicly Available Models
2. Models/Query Router Accessible
3. Path to Local Deployment/Self-Hosting
4. Pieces to Compose Complex Systems

# Query Router Access

&lt;s&gt;[INST]&lt;&lt;SYS&gt;&gt;

{{system\_message}}

&lt;&lt;/SYS&gt;&gt;

{{instruction}} [/INST] {{primer}}

{

	“context”:{{context}}

	“model”:“query”/”doc”

}

<!-- image -->

{

"messages": [{

  "content": "...",

  "role": "system"

},{

  "content": "...",

  "role": "user"

}],

"model": ”mixtral”,

"temperature": 0.2,

"top\_p": 0.7,

"max\_tokens": 1024,

"stream": True

}

<!-- image -->

SDXL

E5

Mixtral

<!-- image -->

Dalle-3

Embed

GPT4

<!-- image -->

https://kubernetes.io/

<!-- image -->

Facilitate

Monitor

Optimize

Load Balance

/chat/completions

<!-- image -->

# Full Deployment Stack

<!-- image -->

https://developer.nvidia.com/blog/simplifying-ai-inference-in-production-with-triton/

SDXL

E5

Mixtral

VideoGen

RIVA

NIMs

NeMo Retriever

Custom Server

General Chatbot

Document

Copilot

Image Generator

K8s

Azure

/AWS

Org

Cluster

TensorRT-LLM / vLLM

Your Device

Remote Host

Jupyter Notebook Server

Frontend

LLMs

Docker

Router

<!-- image -->

<!-- image -->

:8090

:88

# Our Environment

Jupyter Notebooks + Frontend

- LLM Client

<!-- image -->

<!-- image -->

:9000

<!-- image -->

‹#›

<!-- image -->

# NVIDIA Foundation Model Endpoints

NVIDIA 

GPU CLOUD

<!-- image -->

<!-- image -->

# NVIDIA Foundation Model Endpoints

NVIDIA 

GPU CLOUD

<!-- image -->

<!-- image -->

# NVIDIA Foundation Model Endpoints

NVIDIA 

GPU CLOUD

<!-- image -->

<!-- image -->

<!-- image -->

# NVIDIA Foundation Model Endpoints

<!-- image -->

NVIDIA 

GPU CLOUD

<!-- image -->

<!-- image -->

# From Raw Requests to LangChain Model

<!-- image -->

&lt;s&gt;[INST]&lt;&lt;SYS&gt;&gt;

{{system\_message}}

&lt;&lt;/SYS&gt;&gt;

{{instruction}} [/INST] {{primer}}

{

	“context”:{{context}}

	“model”:“query”/”doc”

}

Facilitate

Monitor

Optimize

# From Raw Requests to LangChain Model

<!-- image -->

&gt; llm(“Hello world”)

  ChatMessage(content=”hello”)

&gt; embedder(“Hello”)

  [0.4535437, 0.0800435, ...]

Query Router

<!-- image -->

<!-- image -->

<!-- image -->

Mistral

E5

Llama

# LLM Interfaces

The Whole Stack

<!-- image -->

Scaled

Function Deployment Solution

<!-- image -->

<!-- image -->

api.nvcf.

nvidia.com

/v2/nvcf

<!-- image -->

‹#›

# LLM Interfaces

The Whole Stack

<!-- image -->

Scaled

Function Deployment Solution

<!-- image -->

<!-- image -->

v1/models

v1/completions

v1/chat/completions

v1/embeddings

integrate.api.nvidia.com

health.api.nvidia.com/

ai.api.nvidia.com/

api.nvcf.

nvidia.com

/v2/nvcf

v1/models

v1/completions

v1/chat/completions

v1/embeddings

api.openai.com

<!-- image -->

‹#›

# LLM Interfaces

The Whole Stack

<!-- image -->

Scaled

Function Deployment Solution

<!-- image -->

<!-- image -->

v1/models

v1/completions

v1/chat/completions

v1/embeddings

integrate.api.nvidia.com

health.api.nvidia.com/

ai.api.nvidia.com/

api.nvcf.

nvidia.com

/v2/nvcf

OpenAI

NVIDIABase

v1/models

v1/completions

v1/chat/completions

v1/embeddings

api.openai.com

<!-- image -->

‹#›

# LLM Interfaces

The Whole Stack

<!-- image -->

Scaled

Function Deployment Solution

<!-- image -->

<!-- image -->

v1/models

v1/completions

v1/chat/completions

v1/embeddings

integrate.api.nvidia.com

health.api.nvidia.com/

ai.api.nvidia.com/

api.nvcf.

nvidia.com

/v2/nvcf

OpenAI

NVIDIABase

<!-- image -->

v1/models

v1/completions

v1/chat/completions

v1/embeddings

api.openai.com

<!-- image -->

‹#›

# LLM Interfaces

The Whole Stack

<!-- image -->

Scaled

Function Deployment Solution

<!-- image -->

<!-- image -->

v1/models

v1/completions

v1/chat/completions

v1/embeddings

integrate.api.nvidia.com

health.api.nvidia.com/

ai.api.nvidia.com/

api.nvcf.

nvidia.com

/v2/nvcf

OpenAI

NVIDIABase

<!-- image -->

ChatOpenAI

v1/models

v1/completions

v1/chat/completions

v1/embeddings

api.openai.com

ChatNVIDIA

<!-- image -->

‹#›

# Building RAG Agents with LLMs

Part 3: Intro to LangChain

# LangChain Structure

<!-- image -->

# Chain Building

LLM

Output

Input

<!-- image -->

Just The LLM

‹#›

# Chain Building

LLM

Output

Input

<!-- image -->

LLM

Hey There

AIMessage(“Hello”)

Just The LLM

‹#›

# Chain Building

Prompt

LLM

Output

Input

<!-- image -->

Prompt

Context: {context}

Input: {input}

LLM

{‘input’ : “Hey There”}

Hey There

AIMessage(“Hello”)

Simple Prompt+LLM Chain

‹#›

# Chain Building

Prompt

LLM

Output

Input

<!-- image -->

Prompt

Context: {context}

Input: {input}

LLM

{‘input’ : “Hey There”}

Hey There

AIMessage(“Hello”)

chain = prompt | chat

chain.invoke(inputs)

Simple Prompt+LLM Chain

‹#›

# Chain Building

Prompt

LLM

Output

Input

<!-- image -->

Prompt

Context: {context}

Input: {input}

LLM

{‘input’ : “Hey There”}

Hey There

AIMessage(“Hello”)

chain = prompt | chat

chain.invoke(inputs)

## StrOutputParser()

def get\_content(value):

	return getattr(value, “content”, value)

| get\_content

Simple Prompt+LLM Chain

‹#›

# Chain Building

Prompt

LLM

Output

Input

<!-- image -->

chain = prompt | chat | StrOutputParser()

Invoking Runnables

‹#›

# Chain Building

Prompt

LLM

Output

Input

<!-- image -->

chain = prompt | chat | StrOutputParser()

msg = chain.invoke(...)

for token in chain.stream(...)

Invoking Runnables

‹#›

# Chain Building

External

Prompt

LLM

Output

Input

<!-- image -->

Building Information Pipelines

‹#›

# Chain Building

LLM

Prompt

<!-- image -->

<!-- image -->

Output

LLM

<!-- image -->

If/Else

Database

<!-- image -->

Prompt

<!-- image -->

<!-- image -->

<!-- image -->

Prompt

<!-- image -->

<!-- image -->

Input

<!-- image -->

<!-- image -->

External

Internal

Prompt

LLM

Output

Input

<!-- image -->

<!-- image -->

Output

Building Information Pipelines

‹#›

# LangChain Extended Ecosystem

<!-- image -->

https://github.com/langchain-ai/langchain/tree/master

‹#›

Your Device

Remote Host

Jupyter Notebook Server

Frontend

LLMs

Docker

Router

<!-- image -->

<!-- image -->

:8090

:88

# Our Environment

Jupyter Notebooks + Frontend

- LLM Client

<!-- image -->

<!-- image -->

:9000

<!-- image -->

:9012

‹#›

# Building RAG Agents with LLMs

Part 4: Running State Chain

# Chain Building

LLM

Prompt

<!-- image -->

<!-- image -->

Output

LLM

<!-- image -->

If/Else

Database

<!-- image -->

<!-- image -->

Output

Prompt

<!-- image -->

<!-- image -->

<!-- image -->

Prompt

<!-- image -->

<!-- image -->

Input

<!-- image -->

<!-- image -->

External

Internal

Off-Topic

Needs Lookup

No Lookup

What to do?

Recall Our Assumptions

‹#›

# Chain Building

Prompt

Classify Sentence

LLM

Input

Topic

inputs = {‘input’ : ‘sentence’}

cls\_chain = cls\_prompt | llm

topic = cls\_chain.invoke(inputs)

Towards Running State

‹#›

# Chain Building

Prompt

Classify Sentence

LLM

Input

Prompt

Generate Sentence

LLM

Output

Topic

inputs = {‘input’ : ‘sentence’}

cls\_chain = cls\_prompt | llm

topic = cls\_chain.invoke(inputs)

gen\_chain = out\_prompt | llm

out\_chain = cls\_chain | gen\_chain

for token in out\_chain.stream(inputs):

    print(token, end=””)

Towards Running State

‹#›

# Chain Building

Prompt

Classify Sentence

LLM

Input

inputs = {‘input’ : ‘sentence’}

new\_sentence = out\_chain.invoke(inputs)

Prompt

Combine Sentences

LLM

<!-- image -->

Prompt

Generate Sentence

LLM

Input

Output

Output

Towards Running State

‹#›

# Chain Building

Prompt

Classify Sentence

LLM

Input

inputs = {‘input’ : ‘sentence’}

new\_sentence = out\_chain.invoke(inputs)

Prompt

Combine Sentences

LLM

<!-- image -->

Prompt

Generate Sentence

LLM

Input

inputs.update({‘new’ : ‘new\_sentence’})

for token in merge\_chain.stream(inputs):

    print(token, end=””)

Output

Output

Towards Running State

‹#›

# Chain Building

Prompt

Classify Sentence

LLM

Input

out\_chain = RunnableAssign({

new\_sentence : cls\_chain | gen\_chain

}) | merge\_chain

Prompt

Combine Sentences

LLM

<!-- image -->

Prompt

Generate Sentence

LLM

Input

Output

Output

RunnableAssign

Branch Chain

Branch Chain

Towards Running State

‹#›

# Typical Running State

<!-- image -->

<!-- image -->

<!-- image -->

Regular Fibonacci w/ While Loop

‹#›

# Running State Chain Components

<!-- image -->

Towards LCEL While Loop 

State

n = 8

fib = [0,1]

‹#›

# Running State Chain Components

Towards LCEL While Loop 

State

n = 8

fib = [0,1]

<!-- image -->

State

n = 7

fib = [0,1,1]

msg=“Hello”

‹#›

# Running State Chain Components

Towards LCEL While Loop 

State

n = 8

fib = [0,1,1]

[0,1,1,2]

<!-- image -->

‹#›

# Running State Chain Components

Towards LCEL While Loop 

State

n = 8

fib = [0,1,1]

[0,1,1,2]

<!-- image -->

<!-- image -->

State

n = 8

fib = [0,1,1,2]

‹#›

<!-- image -->

# Running State Chain Components

Towards LCEL While Loop 

State

n = 8

fib = [0,1,1]

<!-- image -->

<!-- image -->

State

n = 8

fib = [0,1,1,2]

‹#›

State

n = 8

fib = 

[0,1,1,2, 3]

# Final Running State Loop

<!-- image -->

‹#›

# Typical Running State Loop

<!-- image -->

<!-- image -->

Comparing Typical with Running State

‹#›

# Final Running State Loop

RunnableBranch

RunnableLambda

<!-- image -->

RunnableAssign

Big Picture

‹#›

# Airline Chatbot

LLM

<!-- image -->

Prompt

Update Knowledge

Input

DB Lookup

Prompt

Format

LLM

<!-- image -->

Prompt

Update Knowledge

History

User

+Knowledge Base

+Customer Info

+Response

‹#›

# Modern Chain Paradigms

Prompt

LLM

Input

Unstructured Generation

Towards Powerful Running State

‹#›

# Modern Chain Paradigms

Prompt

LLM

Input

Unstructured Generation

Towards Powerful Running State

Prompt

Code

LLM

Structured Retrieval

Environment

<!-- image -->

https://python.langchain.com/docs/use\_cases/qa\_structured/sql

‹#›

# Modern Chain Paradigms

Prompt

LLM

Input

Unstructured Generation

Towards Powerful Running State

Prompt

Code

LLM

Structured Retrieval

Environment

Prompt

LLM

Guided Generation

Grammar

/Schema

{“first\_name” : “Jane”,

	“last\_name”  : “Doe”,

	“confirmation” : -1}

{“first\_name” : “unknown”,

	“last\_name”  : “unknown”,

	“confirmation” : -1}

Update given new info:

“Sure, my name is Jane Doe!”

LLM

‹#›

# Modern Chain Paradigms

Prompt

LLM

Input

Unstructured Generation

Towards Powerful Running State

Prompt

Code

LLM

Structured Retrieval

Environment

Prompt

LLM

Guided Generation

Grammar

/Schema

Prompt

LLM

Tool Choice

Tooling

Tool

Tool Schema

‹#›

# Final Objective

Knowledge Base + Running State Chain

<!-- image -->

{

	“first\_name” : “Jane”,

	“last\_name”  : “Doe”,

	“confirmation” : -1,

	...

}

{

	“first\_name” : “unknown”,

	“last\_name”  : “unknown”,

	“confirmation” : -1,

	...

}

Update given new info:

“Sure, my name is Jane Doe!”

LLM

‹#›

# Airline Chatbot

LLM

<!-- image -->

Prompt

Update Knowledge

Input

DB Lookup

Prompt

Format

LLM

<!-- image -->

Prompt

Update Knowledge

+Knowledge Base

+Customer Info

+Response

‹#›

# Airline Chatbot

LLM

<!-- image -->

Prompt

Update Knowledge

Input

DB Lookup

Prompt

Format

LLM

<!-- image -->

Prompt

Update Knowledge

+Knowledge Base

+Customer Info

+Response

‹#›

# Airline Chatbot

LLM

<!-- image -->

Prompt

Update Knowledge

Input

DB Lookup

Prompt

Format

LLM

<!-- image -->

Prompt

Update Knowledge

History

User

+Knowledge Base

+Customer Info

+Response

‹#›

# Building RAG Agents with LLMs

Part 5: Working with Documents

# Modern Chain Paradigms

Prompt

LLM

Input

Unstructured Generation

Towards Powerful Running State

Prompt

Code

LLM

Structured Retrieval

Environment

Prompt

LLM

Guided Generation

Grammar

/Schema

Prompt

LLM

Tool Choice

Tooling

Tool

Tool Schema

‹#›

# Document Reasoning

LLM

<!-- image -->

Prompt

Context Question

Sure, I can answer by referring this blog post!

<!-- image -->

<!-- image -->

https://developer.nvidia.com/blog/

‹#›

# Document Reasoning

Company

Database

<!-- image -->

LLM

<!-- image -->

Prompt

Context Question

Sure, I can answer by referring this blog post!

‹#›

# Document Reasoning

Company

Database

<!-- image -->

Your Device

Local files

LLM

<!-- image -->

Prompt

Context Question

Sure, I can answer by referring this blog post!

‹#›

# Document Reasoning

https://js.langchain.com/docs/use\_cases/question\_answering/

Company

Database

<!-- image -->

Your Device

Local files

LLM

<!-- image -->

Prompt

Context Question

Sure, I can answer by referring this blog post!

<!-- image -->

‹#›

# Document Reasoning

https://js.langchain.com/docs/use\_cases/question\_answering/

Company

Database

<!-- image -->

Your Device

Local files

LLM

<!-- image -->

Prompt

Context……………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………Question

I forgot the instructions, but I can still say things

<!-- image -->

https://d2l.ai/

<!-- image -->

‹#›

# Chunking 

<!-- image -->

https://arxiv.org/pdf/2307.09288.pdf

<!-- image -->

‹#›

# Document Stuffing 

<!-- image -->

https://arxiv.org/pdf/2307.09288.pdf

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

LLM

<!-- image -->

Prompt

(DOCS 1-4)

Use Them Please

Question

Oh yes, the intro tells me everything! You see…

‹#›

# Map Reduce Chain

https://arxiv.org/pdf/2307.09288.pdf

LLM

<!-- image -->

Prompt

Useful Info

<!-- image -->

<!-- image -->

<!-- image -->

 ⠇

Smaller Chunk

Smaller Chunk

Smaller Chunk

‹#›

# Refinement Chain

https://arxiv.org/pdf/2307.09288.pdf

LLM

<!-- image -->

Prompt

Useful Info

Summary

 for doc in docs:

	   	yield doc

Top 10 Chunks

Main Ideas

‹#›

# Knowledge Graph Construction

https://blog.langchain.dev/using-a-knowledge-graph-to-implement-a-devops-rag-application/

LLM

<!-- image -->

Prompt

Chapter Logic

Chapters

 for doc in docs:

	   	yield doc

Abstractions

Abstraction Main Ideas

Names

Identity Key Points

LLM

<!-- image -->

Prompt

Useful Constructs

LLM

<!-- image -->

Prompt

Character Information

Per-Chapter Main Ideas

‹#›

# Knowledge Graph Traversal

https://blog.langchain.dev/using-a-knowledge-graph-to-implement-a-devops-rag-application/

LLM

<!-- image -->

Prompt

Find Info

Chapters

Abstractions

Abstraction Main Ideas

Per-Chapter Main Ideas

Names

Identity Key Points

LLM

<!-- image -->

Prompt

Use Info

Sure! From the chapter on Birds, I can tell you…

How does Flying work according to your book?

‹#›

# Refinement Chain

https://arxiv.org/pdf/2307.09288.pdf

LLM

<!-- image -->

Prompt

Useful Info

Summary

 for doc in docs:

	   	yield doc

Top 10 Chunks

Main Ideas

Your Assignment

‹#›

# Refinement Chain

https://arxiv.org/pdf/2307.09288.pdf

LLM

<!-- image -->

Prompt

Useful Info

Summary

 for doc in docs:

	   	yield doc

RunnableLambda

Your Assignment

‹#›

# Optional Tangent: LangGraph

<!-- image -->

https://python.langchain.com/docs/langgraph/

https://blog.langchain.dev/langgraph-multi-agent-workflows/

<!-- image -->

‹#›

# Refinement Chain

https://arxiv.org/pdf/2307.09288.pdf

LLM

<!-- image -->

Prompt

Useful Info

Summary

 for doc in docs:

	   	yield doc

RunnableLambda

Your Assignment

‹#›

# Building RAG Agents with LLMs

Part 6: Embedding Model for Retrieval

# Knowledge Graph Traversal

https://blog.langchain.dev/using-a-knowledge-graph-to-implement-a-devops-rag-application/

LLM

<!-- image -->

Prompt

Find Info

Chapters

Abstractions

Abstraction Main Ideas

Per-Chapter Main Ideas

Names

Identity Key Points

LLM

<!-- image -->

Prompt

Use Info

Sure! From the chapter on Birds, I can tell you…

How does Flying work according to your book?

‹#›

# Modern Chain Paradigms

Prompt

LLM

Input

Unstructured Generation

Towards Powerful Running State

Prompt

Code

LLM

Structured Retrieval

Environment

Prompt

LLM

Unstructured Retrieval

Unstructured Retrieval

???

‹#›

# Modern Chain Paradigms

Prompt

LLM

Input

Unstructured Generation

Towards Powerful Running State

Prompt

Code

LLM

Structured Retrieval

Environment

Prompt

LLM

Unstructured Retrieval

Unstructured Retrieval

Vector

Database

‹#›

# Retrieval-Augmented Generation

Pulling in Information from a Database

https://cs.stanford.edu/~myasu/blog/racm3/ 

<!-- image -->

https://www.nvidia.com/en-us/training/instructor-led-workshops/rapid-application-development-using-large-language-models/

# Transformer Architecture

Primary Backbone of LLMs

<!-- image -->

https://arxiv.org/pdf/1706.03762.pdf

Element-Wise Feed-Forward

Element-Wise Feed-Forward

Sequence Attention Interface

Element-Wise Feed-Forward

Sequence Attention Interface

Element-Wise Feed-Forward

‹#›

# Transformer Architecture

<!-- image -->

Autoregressing vs Embedding Flavors

‹#›

# Transformer Architecture

<!-- image -->

<!-- image -->

Autoregressing vs Embedding Flavors

‹#›

# Retrieval QA Embedding

Asymmetric Query/Document Model

<!-- image -->

<!-- image -->

https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/nvolve-29k/api

# Embedding and Comparing

Querying for Semantically Similar Entries

https://developer.nvidia.com/blog/accelerating-vector-search-using-gpu-powered-indexes-with-rapids-raft/

High-performance computing.

Happy Holidays!

DLSS Gaming Statistics

Any cool video games lately?

Biological vision structure

What’s with GPUs these days?

Mitochondria, powerhouse

<!-- image -->

# Embedding and Comparing

Querying for Semantically Similar Entries

https://developer.nvidia.com/blog/accelerating-vector-search-using-gpu-powered-indexes-with-rapids-raft/

High-performance computing.

Happy Holidays!

DLSS Gaming Statistics

Any cool video games lately?

Biological vision structure

What’s with GPUs these days?

Mitochondria, powerhouse

<!-- image -->

# Embedding and Comparing

Querying for Semantically Similar Entries

https://developer.nvidia.com/blog/accelerating-vector-search-using-gpu-powered-indexes-with-rapids-raft/

High-performance computing.

Happy Holidays!

DLSS Gaming Statistics

Any cool video games lately?

Biological vision structure

What’s with GPUs these days?

Mitochondria, powerhouse

<!-- image -->

# Embedding Spaces

Learned Semantically Dense Values

https://ichko.github.io/visualizing-vae-with-efemarai 

<!-- image -->

<!-- image -->

<!-- image -->

https://queenscompsci.wordpress.com/2018/03/04/interpolation-in-the-latent-space-of-variational-autoencoders/

https://en.wikipedia.org/wiki/T-distributed\_stochastic\_neighbor\_embedding

# Language Embedding Schemes

Bi-Encoder versus Cross-Encoder

Encoder

Classifier

Passage 1

Passage 2

Bi-Encoder

Cross-Encoder

Encoder

Encoder

u

v

Cosine-Similarity

Passage 1

Passage 2

https://www.sbert.net/examples/applications/cross-encoder/README.html

Reranker

Embedding

# Language Embedding Schemes

Symmetric versus Asymmetric

Encoder 1

Encoder 1

u

v

Cosine-Similarity

Passage 1

Passage 2

Symmetric

Asymmetric/Generalized

<!-- image -->

https://cs.stanford.edu/~myasu/blog/racm3/ 

https://www.sbert.net/examples/applications/cross-encoder/README.html

# Language Embedding Schemes

Generalized Definition

https://cs.stanford.edu/~myasu/blog/racm3/ 

<!-- image -->

<!-- image -->

<!-- image -->

# Embedding and Comparing

Querying for Semantically Similar Entries

https://developer.nvidia.com/blog/accelerating-vector-search-using-gpu-powered-indexes-with-rapids-raft/

High-performance computing.

Happy Holidays!

DLSS Gaming Statistics

Any cool video games lately?

Biological vision structure

What’s with GPUs these days?

Mitochondria, powerhouse

<!-- image -->

# Building RAG Agents with LLMs

Part 6.4: Guardrails

<!-- image -->

# Embedding and Comparing

Querying for Semantically Similar Entries

https://developer.nvidia.com/blog/accelerating-vector-search-using-gpu-powered-indexes-with-rapids-raft/

High-performance computing.

Happy Holidays!

DLSS Gaming Statistics

Any cool video games lately?

Biological vision structure

What’s with GPUs these days?

Mitochondria, powerhouse

<!-- image -->

# Semantic Guardrails

LLM

<!-- image -->

Prompt

Answer Please Question

How’s the weather?

No. I shouldn’t answer that

Classifier

Branch

Prompt

Don’t Answer question

Tell me about GPUs

Illegal Topics

What’s an LLM Service

What’s a good game with RTX?

Irrelevant Questions

‹#›

# Embedding Classification

Classifying with embeddings

https://developer.nvidia.com/blog/accelerating-vector-search-using-gpu-powered-indexes-with-rapids-raft/

Illegal Topics

Irrelevant Questions

Tell me about GPUs

What’s an LLM Service

What’s a good game with RTX?

<!-- image -->

https://developer.nvidia.com/blog/accelerating-vector-search-using-gpu-powered-indexes-with-rapids-raft/

Illegal Topics

Irrelevant Questions

Tell me about GPUs

What’s an LLM Service

# Embedding Classification

Classifying with embeddings

What’s a good game with RTX?

<!-- image -->

https://developer.nvidia.com/blog/accelerating-vector-search-using-gpu-powered-indexes-with-rapids-raft/

Illegal Topics

Irrelevant Questions

Tell me about GPUs

What’s an LLM Service

What’s a good game with RTX?

0 or 1?

<!-- image -->

# Embedding Classification

Classifying with embeddings

Classification Head

# Semantic Guardrails

LLM

<!-- image -->

Prompt

Answer Please Question

How’s the weather?

No. I shouldn’t answer that

Classifier

Branch

Prompt

Don’t Answer question

Tell me about GPUs

Illegal Topics

What’s an LLM Service

What’s a good game with RTX?

Irrelevant Questions

‹#›

# Building RAG Agents with LLMs

Part 7: Document Retrieval with Vector Databases

<!-- image -->

# Embedding and Comparing

Querying for Semantically Similar Entries

https://developer.nvidia.com/blog/accelerating-vector-search-using-gpu-powered-indexes-with-rapids-raft/

High-performance computing.

Happy Holidays!

DLSS Gaming Statistics

Any cool video games lately?

Biological vision structure

What’s with GPUs these days?

Mitochondria, powerhouse

<!-- image -->

# Retrieval-Augmented Generation

Pulling in Information from a Database

https://cs.stanford.edu/~myasu/blog/racm3/ 

<!-- image -->

https://www.nvidia.com/en-us/training/instructor-led-workshops/rapid-application-development-using-large-language-models/

# Retrieval QA Embedding

Asymmetric Query/Document Model

<!-- image -->

<!-- image -->

https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/nvolve-29k/api

‹#›

# Integrating a Vector Store

<!-- image -->

‹#›

# Integrating a Vector Store

<!-- image -->

https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/

<!-- image -->

‹#›

<!-- image -->

VDB

# Integrating a Vector Store

<!-- image -->

https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/

‹#›

<!-- image -->

VDB

# Integrating a Vector Store

<!-- image -->

https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/

Retriever

‹#›

<!-- image -->

VDB

# Integrating a Vector Store

<!-- image -->

https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/

Retriever

&lt;metadata&gt; 

your name is NVBot…

&lt;conversation&gt; 

Hey, my name is Jane

&lt;wikipedia&gt; 

…a name is a term…

‹#›

VDB

# Retrieval Reordering/Selection

Retriever

&lt;metadata&gt; 

your name is NVBot…

&lt;conversation&gt; 

Hey, my name is Jane

&lt;wikipedia&gt; 

…a name is a term…

What’s

my name?

Reranker

LongContextReorder

‹#›

VDB

# Query Augmentation

Retriever

What’s

my name?

LLM

<!-- image -->

Prompt

Rephrase as Question

LLM

<!-- image -->

Prompt

Rephrase as Hypothesis

‹#›

VDB

# RAG Fusion

Retriever

What’s

my name?

LLM

<!-- image -->

Prompt

Rephrase as Question

LLM

<!-- image -->

Prompt

Rephrase as Hypothesis

Reranker

‹#›

<!-- image -->

# Integrating a Vector Store

<!-- image -->

https://faiss.ai/

‹#›

<!-- image -->

# Integrating a Local Vector Store

https://faiss.ai/

Your Device

Local Host

Milvus Standalone

Frontend

Jupyter Notebook

FAISS

‹#›

# Integrating a Local Vector Store

Your Device

Local Host

Milvus Standalone

Frontend

Jupyter Notebook

FAISS

&lt;s&gt;[INST]&lt;&lt;SYS&gt;&gt;

{{system\_message}}

&lt;&lt;/SYS&gt;&gt;

{{instruction}} [/INST] {{primer}}

{

	“context”:{{context}}

	“model”:“query”/”doc”

}

<!-- image -->

Query Router

<!-- image -->

<!-- image -->

Mistral

E5

Llama

doc1

doc2

doc3

‹#›

<!-- image -->

# GPU-Accelerating Vector Stores

https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/

<!-- image -->

https://developer.nvidia.com/blog/accelerating-vector-search-using-gpu-powered-indexes-with-rapids-raft/

‹#›

Milvus Standalone

Milvus Cluster

<!-- image -->

https://developer.nvidia.com/blog/making-data-science-teams-productive-kubernetes-rapids/

Jupyter Notebook

Server      .

FAISS

<!-- image -->

# Compute Scale Progression

‹#›

# Simple Conversation RAG Setup

LLM

<!-- image -->

Prompt

History

Context

Question

It’s Jane!

What’s My Name?

VDB

Retriever

Hello Jane! How are you?

Hello! My name is Jane

Retriever

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

VDB

‹#›

# Simple RAG Agents

LLM

<!-- image -->

Prompt

Question

How does RAG work?

Sure! According to the paper…

Prompt

Context

Question

Classifier

Branch

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

Retriever

VDB

‹#›

LLM

<!-- image -->

Prompt

Context

Question

LLM Data Retriever

# Proper RAG Agent

Tool-Selection Agents

Final Answer

- History

LLM

<!-- image -->

Prompt

Question

Answer

LLM

<!-- image -->

Prompt

What Should I do?

How does RAG work?

<!-- image -->

<!-- image -->

<!-- image -->

TOOLSET

‹#›

# Building RAG Agents with LLMs

Part 8: RAG Evaluation

# LLM-As-A-Judge

Pipeline Evaluation

LLM

<!-- image -->

Prompt

Context

Question

Retriever

How does RAG work?

According to my resources…

RAG Pipeline

‹#›

LLM

<!-- image -->

Prompt

Context

Question

Retriever

How does RAG work?

According to my resources…

# LLM-As-A-Judge

Pipeline Evaluation

LLM

<!-- image -->

Prompt

Is This Good?

LLM

<!-- image -->

Prompt

Facilitate Testing

RAG Pipeline

Evaluation Pipeline

Functions

Functions

‹#›

# Our Evaluation Chain Components

Synthetic Generation

LLM

<!-- image -->

Prompt

Ask+Answer

Q: How Does RAG Work?

A: From these documents…

VDB

Doc1

Doc2

‹#›

# Our Evaluation Chain Components

RAG Pipeline Sample

LLM

<!-- image -->

Prompt

Context

Question

How does RAG work?

According to my resources…

Retriever

LLM

<!-- image -->

Prompt

Ask+Answer

Q: How Does RAG Work?

A: From these documents…

VDB

Doc1

Doc2

‹#›

LLM

<!-- image -->

Prompt

Ask+Answer

Q: How Does RAG Work?

A: From these documents…

VDB

Doc1

Doc2

LLM

<!-- image -->

Prompt

Context

Question

Retriever

How does RAG work?

According to my resources…

LLM

<!-- image -->

Prompt

Which Is Better?

# Our Evaluation Chain Components

Ground Truth vs Our Pipeline

[1] Bot 2 Better

‹#›

LLM

<!-- image -->

Prompt

Ask+Answer

Q: How Does RAG Work?

A: From these documents…

Doc1

Doc2

LLM

<!-- image -->

Prompt

Which Is Better?

# Our Evaluation Chain Components

Ground Truth vs Our Pipeline

[1] Bot 2 Better

[1]

[0]

[1]

[0]

[1]

4/6

LLM

<!-- image -->

Prompt

Context

Question

How does RAG work?

According to my resources…

Retriever

VDB

‹#›

LLM

<!-- image -->

VDB

- Embedder

LLM

<!-- image -->

Prompt

Context

Question

Retriever

{question}

# General Evaluation Chain Components

Multiple Metrics at Once

RagasEvaluatorChain

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

https://github.com/explodinggradients/ragas

<!-- image -->

‹#›

<!-- image -->

# RAG Evaluation with 

https://arxiv.org/pdf/2307.09288.pdf

<!-- image -->

Quantifying System Goodness with LLM-as-a-Judge

<!-- image -->

<!-- image -->

Control

Sources

Input

RagasEvaluatorChain

<!-- image -->

# RAG Evaluation with 

https://arxiv.org/pdf/2307.09288.pdf

<!-- image -->

Quantifying System Goodness with LLM-as-a-Judge

<!-- image -->

<!-- image -->

<!-- image -->

https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/\_faithfulness.py

# Evaluator Agent

Pipeline Evaluation

Final Answer

- History

LLM

<!-- image -->

Prompt

Is This Good?

LLM

<!-- image -->

Prompt

What Should I do?

<!-- image -->

LLM

<!-- image -->

Prompt

Ask some questions

TOOLSET

Chain to Evaluate

‹#›

LLM

<!-- image -->

Prompt

Ask+Answer

Q: How Does RAG Work?

A: From these documents…

Doc1

Doc2

LLM

<!-- image -->

Prompt

Which Is Better?

# Our Evaluation Chain Components

Ground Truth vs Our Pipeline

[1] Bot 2 Better

[1]

[0]

[1]

[0]

[1]

4/6

LLM

<!-- image -->

Prompt

Context

Question

How does RAG work?

According to my resources…

Retriever

VDB

‹#›

# Evaluate RAG In The Frontend

Your Device

Remote Host

Frontend

NVIDIA 

GPU CLOUD

<!-- image -->

Jupyter Notebook

<!-- image -->

<!-- image -->

:8090

RAG

Final Assessment

‹#›

# Congratulations!

AND THANK YOU SO MUCH!

<!-- image -->