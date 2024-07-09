namespace PerceptivePyro.GPT;

using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using F = nn.functional;


public class StreamingGPTModel : GPTModel
{
    private readonly Tokenizer encoding;
    private Tensor idx;
    private int block_size;
    private double temperature;
    private int? top_k;
    private readonly string device;
    private int current_length;

    public StreamingGPTModel(GPTConfig config, Tokenizer encoding, double temperature, int? top_k = 1, string device = "cpu") : base(config)
    {
        this.top_k = top_k;
        this.device = device;
        this.encoding = encoding;
        this.temperature = temperature;
        this.block_size = config.block_size;
    }

    public async IAsyncEnumerable<int> StreamAsync(IAsyncEnumerable<int> stream)
    {
        // We start with a 100ms buffer,
        // this allows up to 100ms to elapse during inference before we process the next buffer.
        // This buffer will adjust in size.
        int timeBufferSize = 100; 
        var s = Stopwatch.StartNew();
        var tokenBuffer = new List<int>();
        await foreach (var token in stream)
        {
            if (s.ElapsedMilliseconds < timeBufferSize)
            {
                if (this.encoding.Decode(new[] { token }) is not "<PAD>" or "<UNK>")
                {
                    // If we are within the window, keep adding to the buffer.                
                    tokenBuffer.Add(token);                    
                }
            }
            else
            {
                if (tokenBuffer.Any())
                {
                    var gpt_context = torch.as_tensor(tokenBuffer, dtype: @long, device: this.device).reshape(1, tokenBuffer.Count);
                    var output = this.GenerateNextTokens(gpt_context);
                    foreach (var idx in output[0].data<long>())
                    {
                        yield return (int)idx;
                    }

                    s.Restart();
                }
            }
        }
    }

    public Tensor GenerateNextTokens(Tensor newTokens)
    {
        using var no_grad = torch.no_grad();

        // Add the new token to the buffer
        this.idx[0, current_length % block_size] = newTokens;
        current_length++;

        // Create a mask to only consider the valid part of the buffer
        var valid_length = Math.Min(current_length, block_size);
        using var mask = torch.zeros(new long[] { 1, block_size }, dtype: torch.@bool);
        mask[0, ..valid_length] = 1;

        // Apply the mask to the buffer
        using var idx_cond = this.idx.masked_fill(~mask, 0);

        // forward the model to get the logits for the index in the sequence
        var (logits, _) = this.call(idx_cond, null, false);

        // pluck the logits at the final step and scale by desired temperature
        logits = logits[.., valid_length - 1, ..] / this.temperature;

        // optionally crop the logits to only the top k options
        if (this.top_k is int k)
        {
            var top = Math.Min(k, (int)logits.size(-1));
            var (v, top_indexes) = torch.topk(logits, top);
            var lowest = v[.., -1].reshape(1, 1);
            logits.masked_fill_(logits < lowest, float.NegativeInfinity);
        }

        // apply softmax to convert logits to (normalized) probabilities
        using var probs = F.softmax(logits, dim: -1);

        // sample from the distribution
        using var idx_next = torch.multinomial(probs, num_samples: 1);

        // Add the next token to the buffer
        this.idx[0, current_length % block_size] = idx_next;
        current_length++;

        return idx_next;
    }
}
