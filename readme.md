# Index and search images and texts in specified directory

Semantic search engine for local files (text and images). It can index files in directory (recursively) and allows searching via a text query or/and image query. Includes a Gradio web UI for interaction.

Microservice architecture:
- file-discovery-service watches the `/data` directory, pushes file changes to a Redis queue
- main-service consumes directory updates, uses embedding microservice to generate file embeddings, and stores the results in a Qdrant database for Cosine search
- embedding-service hosts CLIP-like model for calculating embeddings from texts and images

## Configure model

Model can be changed in global_config.env. Need to specify model checkpoint (from HuggingFace), embedding size and target model image resolution. Supports models: 
- openai/clip-vit-large-patch14
- jinaai/jina-clip-v1
- jinaai/jina-clip-v2
- google/siglip-base-patch16-224 (and other SigLip models) - currently useless
- google/siglip2-base-patch16-224 (and other SigLip 2 models) - currently useless

Example default config for jinaai/jina-clip-v1:

```
EMBEDDING_MODEL_CHECKPOINT = jinaai/jina-clip-v1
EMBEDDING_SIZE = 768
TARGET_IMAGE_SIZE = 224,224
```

## To start backend:
```
docker compose up -d --build
```
Web UI: http://localhost:8002/ui
Main API docs: http://localhost:8002/docs

## Service API

### Search files

`POST /api/v1/search`

Performs a search. Takes optional text_query (form), image_query (file), and top_n (form) as input. Returns filenames and scores

### Upload files

`POST /api/v1/files`

Uploads one or more files to be indexed. Takes a list of multipart file uploads

### Manually index files

`POST /api/v1/index`

Triggers indexing for a list of file paths already existing in `/data` directory

### Delete files from index

`DELETE /api/v1/index`

Removes files from the search index based on a list of file paths. File is not deleted. It will be reindexed on next app restart
