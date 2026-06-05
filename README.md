ANTHROPIC_BASE_URL="http://localhost:11434" ANTHROPIC_API_KEY="local_dummy_key" ollama launch claude --model gemma4:e4b-mlx

or

export ANTHROPIC_BASE_URL="http://localhost:11434"
export ANTHROPIC_API_KEY="local_dummy_key"

ollama launch claude

unset ANTHROPIC_BASE_URL
unset ANTHROPIC_API_KEY

echo $ANTHROPIC_BASE_URL
echo $ANTHROPIC_API_KEY

```
tree -I 'node_modules|.git' > structure.txt
```

claude --resume ba881658-2a4b-4126-8ca0-f21803c5e8b9
