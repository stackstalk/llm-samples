How to use
----------
1. Bring up the services using docker compose file.

2. Pull the LLM model into the ollama container
docker exec -it ollama bash
ollama pull llama:3.2

3. Configure lnagfuse public key and secret key in the docker compose file by doing a signup in langfuse URL 

4. Langfuse URL: http://localhost:3000/
   LLM app URL: http://localhost:8051/
