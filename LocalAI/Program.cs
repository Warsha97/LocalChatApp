using Microsoft.Extensions.AI;
using OllamaSharp;

// Creating clients for embedding and chat
var ollamaUri = new Uri("http://localhost:11434/");

IChatClient chatClient = new OllamaApiClient(ollamaUri, "phi3:mini");
IEmbeddingGenerator<string, Embedding<float>> embedder =
    new OllamaApiClient(ollamaUri, "nomic-embed-text");

// ── 2. Load & chunk documents ────────────────────────────────────────────────
var docsPath = Path.Combine(AppContext.BaseDirectory, "docs");
var chunks = new List<string>();

if (Directory.Exists(docsPath))
{
    foreach (var file in Directory.GetFiles(docsPath, "*.*")
        .Where(f => f.EndsWith(".txt") || f.EndsWith(".md")))
    {
        var text = await File.ReadAllTextAsync(file);
        chunks.AddRange(ChunkText(text, chunkSize: 300));
    }
}

Console.WriteLine($"Loaded {chunks.Count} chunks from {docsPath}");

// ── 3. Embed all chunks up front ─────────────────────────────────────────────
Console.WriteLine("Embedding chunks, please wait...");
var chunkEmbeddings = new List<(string Chunk, float[] Vector)>();

foreach (var chunk in chunks)
{
    var result = await embedder.GenerateAsync([chunk]);
    chunkEmbeddings.Add((chunk, result[0].Vector.ToArray()));
}

Console.WriteLine("Ready! Ask anything about your docs.\n");

// ── 4. Chat loop ─────────────────────────────────────────────────────────────
var chatHistory = new List<ChatMessage>();

while (true)
{
    Console.WriteLine("Your prompt:");
    var userPrompt = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(userPrompt)) continue;

    // Embed the user question and find the top 3 relevant chunks
    var questionEmbedding = await embedder.GenerateAsync([userPrompt]);
    var questionVector = questionEmbedding[0].Vector.ToArray();

    var topChunks = chunkEmbeddings
        .Select(c => (c.Chunk, Score: CosineSimilarity(questionVector, c.Vector)))
        .OrderByDescending(c => c.Score)
        .Take(3)
        .Select(c => c.Chunk)
        .ToList();

    // Build a context-aware prompt
    var context = string.Join("\n\n", topChunks);
    var augmentedPrompt = $"""
        Use the following context to answer the question.
        If the answer isn't in the context, say so honestly.

        Context:
        {context}

        Question: {userPrompt}
        """;

    chatHistory.Add(new ChatMessage(ChatRole.User, augmentedPrompt));

    Console.WriteLine("AI Response:");
    var response = "";
    await foreach (var item in chatClient.GetStreamingResponseAsync(chatHistory))
    {
        Console.Write(item.Text);
        response += item.Text;
    }

    chatHistory.Add(new ChatMessage(ChatRole.Assistant, response));
    Console.WriteLine("\n");
}

// ── Helpers ───────────────────────────────────────────────────────────────────
static IEnumerable<string> ChunkText(string text, int chunkSize)
{
    var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
    var chunk = new List<string>();
    int count = 0;

    foreach (var word in words)
    {
        chunk.Add(word);
        count++;
        if (count >= chunkSize)
        {
            yield return string.Join(' ', chunk);
            chunk.Clear();
            count = 0;
        }
    }

    if (chunk.Count > 0)
        yield return string.Join(' ', chunk);
}

static float CosineSimilarity(float[] a, float[] b)
{
    float dot = 0, magA = 0, magB = 0;
    for (int i = 0; i < a.Length; i++)
    {
        dot += a[i] * b[i];
        magA += a[i] * a[i];
        magB += b[i] * b[i];
    }
    return dot / (MathF.Sqrt(magA) * MathF.Sqrt(magB));
}