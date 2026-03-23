using System.Collections.Generic;
using System.Diagnostics.Metrics;
using System.Text.Json;
using Microsoft.Extensions.AI;
using OllamaSharp;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Content;

// ── 1. Clients ──────────────────────────────────────────────────────────────
var ollamaUri = new Uri("http://localhost:11434/");

IChatClient chatClient = new OllamaApiClient(ollamaUri, "phi3:mini");
IEmbeddingGenerator<string, Embedding<float>> embedder =
    new OllamaApiClient(ollamaUri, "nomic-embed-text");

// ── 2. Load & chunk documents ────────────────────────────────────────────────
var docsPath = Path.Combine(AppContext.BaseDirectory, "docs");
var cachePath = Path.Combine(AppContext.BaseDirectory, "embeddings.cache.json");
var chunks = new List<string>();

foreach (var file in Directory.GetFiles(docsPath, "*.*")
    .Where(f => f.EndsWith(".txt") || f.EndsWith(".md") || f.EndsWith(".pdf")))
{
    string text;

    if (file.EndsWith(".pdf"))
    {
        text = ExtractTextFromPdf(file);
        Console.WriteLine($"Extracted text from PDF: {Path.GetFileName(file)}");
    }
    else
    {
        text = await File.ReadAllTextAsync(file);
    }

    chunks.AddRange(ChunkText(text, chunkSize: 150));
}

Console.WriteLine($"Loaded {chunks.Count} chunks from {docsPath}");

// ── 3. Embed chunks (or load from cache) ────────────────────────────────────
var chunkEmbeddings = new List<(string Chunk, float[] Vector)>();
var currentHash = await ComputeDocsHash(docsPath);

if (File.Exists(cachePath))
{
    Console.WriteLine("Loading embeddings from cache...");
    var cached = JsonSerializer.Deserialize<List<CachedEmbedding>>(
        await File.ReadAllTextAsync(cachePath));

    var cachedHash = cached?.FirstOrDefault()?.DocsHash ?? "";

    if (cached != null && cachedHash == currentHash)
    {
        chunkEmbeddings = cached.Select(c => (c.Chunk, c.Vector)).ToList();
        Console.WriteLine("Cache loaded successfully.");
    }
    else
    {
        Console.WriteLine("Docs have changed, re-embedding...");
        chunkEmbeddings = await EmbedAndCache(chunks, embedder, cachePath, currentHash);
    }
}
else
{
    Console.WriteLine("No cache found, embedding for the first time...");
    chunkEmbeddings = await EmbedAndCache(chunks, embedder, cachePath, currentHash);
}

// ── 4. Chat loop ─────────────────────────────────────────────────────────────
var chatHistory = new List<ChatMessage>();

while (true)
{
    Console.WriteLine("Your prompt:");
    var userPrompt = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(userPrompt)) continue;

    var questionEmbedding = await embedder.GenerateAsync([userPrompt]);
    var questionVector = questionEmbedding[0].Vector.ToArray();

    var topChunks = chunkEmbeddings
        .Select(c => (c.Chunk, Score: CosineSimilarity(questionVector, c.Vector)))
        .OrderByDescending(c => c.Score)
        .Take(3)
        .Select(c => c.Chunk)
        .ToList();

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
static async Task<List<(string Chunk, float[] Vector)>> EmbedAndCache(
    List<string> chunks,
    IEmbeddingGenerator<string, Embedding<float>> embedder,
    string cachePath, string docsHash)
{
    var results = new List<(string Chunk, float[] Vector)>();
    var toCache = new List<CachedEmbedding>();

    foreach (var chunk in chunks)
    {
        var result = await embedder.GenerateAsync([chunk]);
        var vector = result[0].Vector.ToArray();
        results.Add((chunk, vector));
        toCache.Add(new CachedEmbedding { 
            DocsHash = docsHash, 
            Chunk = chunk, 
            Vector = vector });
    }

    await File.WriteAllTextAsync(cachePath, JsonSerializer.Serialize(toCache));
    Console.WriteLine("Embeddings saved to cache.");
    return results;
}

static async Task<string> ComputeDocsHash(string docsPath)
{
    var files = Directory.GetFiles(docsPath, "*.*")
        .Where(f => f.EndsWith(".txt") || f.EndsWith(".md"))
        .OrderBy(f => f) // consistent order
        .ToList();

    using var sha256 = System.Security.Cryptography.SHA256.Create();
    var allBytes = new List<byte>();

    foreach (var file in files)
    {
        // Include the filename so renames are detected too
        allBytes.AddRange(System.Text.Encoding.UTF8.GetBytes(Path.GetFileName(file)));
        allBytes.AddRange(await File.ReadAllBytesAsync(file));
    }

    var hash = sha256.ComputeHash(allBytes.ToArray());
    return Convert.ToHexString(hash);
}

static string ExtractTextFromPdf(string filePath)
{
    var sb = new System.Text.StringBuilder();

    using var pdf = PdfDocument.Open(filePath);
    foreach (var page in pdf.GetPages())
    {
        // GetWords() preserves reading order better than Letters
        var words = page.GetWords();
        sb.AppendLine(string.Join(" ", words.Select(w => w.Text)));
    }

    return sb.ToString();
}

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

// ── Cache model ───────────────────────────────────────────────────────────────
public class CachedEmbedding
{
    public string DocsHash { get; set; } = "";
    public string Chunk { get; set; } = "";
    public float[] Vector { get; set; } = [];
}
