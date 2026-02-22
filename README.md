How to set up:
1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `ollama pull gemma3:4b`
5. download corresponding models' gguf files with Q4_K_M quant and put them into `models` directory
   1. https://huggingface.co/INSAIT-Institute/MamayLM-Gemma-2-9B-IT-v0.1-GGUF
   2. https://huggingface.co/INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0-GGUF
   3. https://huggingface.co/lapa-llm/lapa-v0.1.2-instruct-GGUF
6. make sure model hashes match the ones below using `shasum -a 256 models/${model_name}.gguf`
7. `ollama create lapa-v0.1.2-q4 -f models/lapa-v0.1.2-Q4.Modelfile`
8. `ollama create mamay-9b-q4 -f models/MamayLM-9B-q4.Modelfile`
9. `ollama create mamay-4b-q4 -f models/MamayLM-4B-q4.Modelfile`

Hash sums of models gguf files:
1. models/lapa-v0.1.2-instruct-Q4_K_M.gguf - 7d97b7f45c71f68ea1b6dade484b9c60ffaed8b5345ea4ea37b112c4660e1862
2. models/MamayLM-Gemma-2-9B-IT-v0.1.Q4_K_M.gguf - aa99487e047604cd8fd47efa2d4eaa959256a0a2d0dfb3d974d83c51949a4f2c
3. models/MamayLM-Gemma-3-4B-IT-v1.0.Q4_K_M.gguf - 82ca9f82e62d1449987016838f2470d467e02fe23b36e84b5e245a08f345d4a3
